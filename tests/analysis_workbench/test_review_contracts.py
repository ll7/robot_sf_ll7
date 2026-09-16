"""Focused contract tests for SREV-01 review contracts (issue #9270)."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench import review_contracts
from robot_sf.analysis_workbench.review_contracts import (
    ReviewContractsValidationError,
    admitted_source_receipt_from_dict,
    component_descriptor_from_dict,
    component_request_canonical_digest,
    component_request_from_dict,
    component_result_from_dict,
    experiment_recipe_canonical_digest,
    experiment_recipe_from_dict,
    load_admitted_source_receipt,
    resolve_admitted_source,
    review_bundle_canonical_digest,
    review_bundle_from_dict,
    run,
    visualization_spec_from_dict,
)

COMMIT = "a" * 40
SHA = "b" * 64
ADMITTED_SOURCE_FIXTURE_DIR = (
    Path(__file__).parents[1] / "fixtures" / "scenario_review" / "admitted_source"
)


def _bundle_doc(**overrides: object) -> dict[str, object]:
    doc: dict[str, object] = {
        "schema_version": "review-bundle.v1",
        "bundle_id": "bundle-0000",
        "episodes": [
            {
                "episode_id": "episode-0000",
                "references": [
                    {
                        "artifact_id": "trace-0000",
                        "uri": "trace-0000.json",
                        "format": "simulation_trace_export.v1",
                        "schema": "simulation_trace_export.v1",
                        "sha256": SHA,
                        "source_commit": COMMIT,
                        "units": "seconds/metres/radians",
                        "coordinate_frame": "world",
                    }
                ],
            }
        ],
    }
    doc.update(overrides)
    return doc


def _admitted_source_fixture(name: str) -> Any:
    return json.loads((ADMITTED_SOURCE_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_review_bundle_accepts_valid_index() -> None:
    bundle = review_bundle_from_dict(_bundle_doc())
    assert bundle.bundle_id == "bundle-0000"
    assert review_bundle_canonical_digest(bundle) == review_bundle_canonical_digest(bundle)


def test_review_bundle_rejects_duplicate_scoped_ids() -> None:
    doc = _bundle_doc()
    episodes = list(doc["episodes"])  # type: ignore[union-attr]
    episodes.append(episodes[0])
    doc["episodes"] = episodes
    with pytest.raises(ReviewContractsValidationError, match="duplicate scoped artifact id"):
        review_bundle_from_dict(doc)


def test_review_bundle_rejects_bad_sha_and_traversal() -> None:
    doc = _bundle_doc()
    ref = doc["episodes"][0]["references"][0]  # type: ignore[index,union-attr]
    ref["sha256"] = "not-a-hash"
    ref["uri"] = "../escape.json"
    with pytest.raises(ReviewContractsValidationError, match="sha256|traversal"):
        review_bundle_from_dict(doc)


def test_schema_validation_error_cap_counts_omission_marker() -> None:
    """Schema errors retain at most cap minus one messages plus the marker."""
    payload = _bundle_doc()
    reference = payload["episodes"][0]["references"][0]  # type: ignore[index,union-attr]
    payload["episodes"][0]["references"] = [  # type: ignore[index,union-attr]
        {
            **reference,
            "artifact_id": f"trace-{index}",
            "sha256": "invalid",
            "source_commit": "invalid",
        }
        for index in range(review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS)
    ]

    with pytest.raises(ReviewContractsValidationError) as error:
        review_bundle_from_dict(payload)

    assert len(error.value.errors) == review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS
    assert (
        len(error.value.errors[:-1]) == review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS - 1
    )
    assert error.value.errors[-1] == "additional validation errors omitted"


def test_semantic_validation_error_cap_counts_omission_marker() -> None:
    """Semantic errors retain at most cap minus one messages plus the marker."""
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "request-0000",
        "component_id": "component-0000",
        "sources": [
            {"artifact_id": f"unsafe/{index}", "uri": "source.json", "format": "fixture"}
            for index in range(review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS + 1)
        ],
        "output_directory": "out",
    }

    with pytest.raises(ReviewContractsValidationError) as error:
        component_request_from_dict(payload)

    assert len(error.value.errors) == review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS
    assert (
        len(error.value.errors[:-1]) == review_contracts.MAX_REVIEW_CONTRACT_VALIDATION_ERRORS - 1
    )
    assert error.value.errors[-1] == "additional validation errors omitted"


def test_visualization_spec_validates_intervals_and_units() -> None:
    doc = {
        "schema_version": "visualization-spec.v1",
        "spec_id": "spec-0000",
        "sources": [{"artifact_id": "trace-0000"}],
        "source_intervals": [{"start_s": 0.0, "end_s": 1.0, "include_terminal_frame": True}],
        "units": "seconds/metres/radians",
    }
    spec = visualization_spec_from_dict(doc)
    assert spec.spec_id == "spec-0000"
    bad = {**doc, "source_intervals": [{"start_s": 1.0, "end_s": 1.0}]}
    with pytest.raises(ReviewContractsValidationError, match="end_s must exceed start_s"):
        visualization_spec_from_dict(bad)
    bad_units = {**doc, "units": "furlongs"}
    with pytest.raises(ReviewContractsValidationError, match="unit_transforms"):
        visualization_spec_from_dict(bad_units)
    bad_finite = {
        **doc,
        "source_intervals": [{"start_s": 0.0, "end_s": float("inf")}],
    }
    with pytest.raises(ReviewContractsValidationError, match="non-finite"):
        visualization_spec_from_dict(bad_finite)


@pytest.mark.parametrize(
    "payload_kind",
    [
        "bundle",
        "visualization",
        "descriptor",
        "request",
        "result",
        "recipe",
        "receipt",
    ],
)
def test_direct_api_envelopes_reject_nested_non_finite(payload_kind: str) -> None:
    """Every untrusted envelope boundary rejects nested NaN and infinity values."""
    if payload_kind == "bundle":
        payload = _bundle_doc()
        payload["episodes"][0]["references"][0]["units"] = {"nested": [float("nan")]}  # type: ignore[index,union-attr]
        validator = review_bundle_from_dict
    elif payload_kind == "visualization":
        payload = {
            "schema_version": "visualization-spec.v1",
            "spec_id": "spec-0000",
            "sources": [{"artifact_id": "trace-0000"}],
            "camera": {"nested": [float("inf")]},
        }
        validator = visualization_spec_from_dict
    elif payload_kind == "descriptor":
        payload = {
            "schema_version": "component-descriptor.v1",
            "component_id": "component-0000",
            "component_version": "1.0.0",
            "supported_input_versions": ["component-request.v1"],
            "output_types": ["component-result.v1"],
            "required_capabilities": [{"nested": float("-inf")}],
        }
        validator = component_descriptor_from_dict
    elif payload_kind == "request":
        payload = _admitted_source_fixture("request.json")
        payload["config"] = {"nested": [float("nan")]}
        with pytest.raises(ReviewContractsValidationError, match="non-finite"):
            component_request_from_dict(payload, source="request.json")
        return
    elif payload_kind == "result":
        payload = {
            "schema_version": "component-result.v1",
            "request_id": "request-0000",
            "component_id": "component-0000",
            "status": "failed",
            "diagnostics": [{"nested": {"value": float("inf")}}],
        }
        validator = component_result_from_dict
    elif payload_kind == "recipe":
        payload = _admitted_source_fixture("recipe.json")
        payload["control_conditions"] = {"nested": [float("-inf")]}
        validator = experiment_recipe_from_dict
    else:
        payload = _admitted_source_fixture("receipt.json")
        payload["source"]["units"] = float("nan")
        validator = admitted_source_receipt_from_dict

    with pytest.raises(ReviewContractsValidationError, match="non-finite"):
        validator(payload)


@pytest.mark.parametrize("field", ["request_id", "component_id"])
def test_component_request_rejects_oversized_identity(field: str) -> None:
    """Direct request construction rejects identities before envelope creation."""
    payload = _admitted_source_fixture("request.json")
    payload[field] = "x" * (review_contracts.MAX_REVIEW_CONTRACT_ID_CHARS + 1)

    with pytest.raises(ReviewContractsValidationError, match="maximum length"):
        component_request_from_dict(payload)


def test_component_result_rejects_oversized_reason() -> None:
    """Source-bound result construction rejects unbounded failure details."""
    payload = {
        "schema_version": "component-result.v1",
        "request_id": "request-0000",
        "component_id": "component-0000",
        "status": "failed",
        "reason": "x" * (review_contracts.MAX_REVIEW_CONTRACT_REASON_CHARS + 1),
    }

    with pytest.raises(ReviewContractsValidationError, match="maximum length"):
        component_result_from_dict(payload, source="result.json")


def test_in_memory_component_config_defers_non_finite_values_to_runner() -> None:
    """In-memory component config stays available for runner-level failure classification."""
    payload = _admitted_source_fixture("request.json")
    payload["config"] = {"nested": [float("nan")]}

    request = component_request_from_dict(payload)

    assert math.isnan(request.config["nested"][0])


def test_in_memory_result_bounds_reason_before_validation() -> None:
    """Internal result details retain their code while satisfying the reason ceiling."""
    payload = {
        "schema_version": "component-result.v1",
        "request_id": "request-0000",
        "component_id": "component-0000",
        "status": "failed",
        "reason": "corrupt_recipe: " + ("detail " * 100),
    }

    result = component_result_from_dict(payload)

    assert result.reason.startswith("corrupt_recipe: ")
    assert len(result.reason) <= review_contracts.MAX_REVIEW_CONTRACT_REASON_CHARS


def test_component_request_rejects_traversal_and_unknown_component_runs_unavailable(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReviewContractsValidationError, match="traversal"):
        component_request_from_dict(
            {
                "schema_version": "component-request.v1",
                "request_id": "r",
                "component_id": "srev01-inspect",
                "sources": [
                    {"artifact_id": "a", "uri": "a.json", "format": "f"},
                ],
                "output_directory": "../escape",
            }
        )
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "r",
            "component_id": "no-such-component",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
            "output_directory": "out",
        }
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported component" in result.reason


def test_run_rejects_missing_capability_and_output_collision(tmp_path: Path) -> None:
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "r",
            "component_id": "srev01-inspect",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
            "output_directory": "out",
            "required_capabilities": ["video-frames"],
        }
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "video-frames" in result.reason
    plain = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "r",
            "component_id": "srev01-inspect",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
            "output_directory": "out",
        }
    )
    first = run(plain, base=tmp_path)
    assert first.status == "complete"
    second = run(plain, base=tmp_path)
    assert second.status == "failed"
    assert "output collision" in second.reason


def test_run_normalizes_non_json_failures_to_component_result(tmp_path: Path) -> None:
    """A valid request that fails while writing still returns the result envelope."""
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "r-nan",
            "component_id": "srev01-inspect",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
            "output_directory": "nan-output",
            "config": {"unserializable": object()},
        }
    )

    result = run(request, base=tmp_path)

    assert result.status == "failed"
    assert result.reason
    assert len(result.reason) <= review_contracts.MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
    parsed = component_result_from_dict(
        {
            "schema_version": "component-result.v1",
            "request_id": result.request_id,
            "component_id": result.component_id,
            "status": result.status,
            "reason": result.reason,
        }
    )
    assert parsed.status == "failed"
    assert not (tmp_path / "nan-output").exists()


def test_run_normalizes_filesystem_failures_to_component_result(tmp_path: Path) -> None:
    """An OS-level output-path failure cannot escape as an exception or huge detail."""
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "r-long-output",
            "component_id": "srev01-capability-report",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
            "output_directory": "x" * 4_096,
        }
    )

    result = run(request, base=tmp_path)

    assert result.status == "failed"
    assert result.reason
    assert len(result.reason) <= review_contracts.MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
    assert not any(path.name == request.output_directory for path in tmp_path.iterdir())


def test_component_result_rejects_partial_with_artifacts() -> None:
    with pytest.raises(ReviewContractsValidationError, match="cannot carry complete status"):
        component_result_from_dict(
            {
                "schema_version": "component-result.v1",
                "request_id": "r",
                "component_id": "c",
                "status": "partial",
                "artifacts": [{"artifact_id": "a", "uri": "a.json", "sha256": SHA}],
            }
        )


def test_experiment_recipe_rejects_duplicate_ids() -> None:
    doc = {
        "schema_version": "experiment-recipe.v1",
        "recipe_id": "recipe-0000",
        "hypothesis": "h",
        "source_identity": {},
        "interventions": [
            {"intervention_id": "i", "factor": "visibility"},
            {"intervention_id": "i", "factor": "delay"},
        ],
        "control_conditions": {},
        "measurements": [{"name": "m", "units": "metres"}],
        "budget": {},
        "stop_rules": ["stop"],
        "preservation_destination": "out",
    }
    with pytest.raises(ReviewContractsValidationError, match="duplicate scoped id"):
        experiment_recipe_from_dict(doc)


@pytest.mark.parametrize("payload_kind", ["request", "recipe"])
def test_retained_configuration_rejects_unicode_surrogates(payload_kind: str) -> None:
    """Retained request and recipe identities cannot carry lone surrogates."""
    if payload_kind == "request":
        payload = _admitted_source_fixture("request.json")
        payload["config"] = {"retained": chr(0xD800)}
        validator = component_request_from_dict
    else:
        payload = _admitted_source_fixture("recipe.json")
        payload["source_identity"]["config_identity"] = chr(0xD800)
        validator = experiment_recipe_from_dict

    with pytest.raises(ReviewContractsValidationError, match="Unicode surrogate"):
        validator(payload)


def test_retained_configuration_accepts_paired_unicode_surrogates() -> None:
    """A directly supplied UTF-16 pair is not mistaken for a lone surrogate."""
    payload = _admitted_source_fixture("request.json")
    payload["config"] = {"retained": chr(0xD83D) + chr(0xDE00)}

    request = component_request_from_dict(payload)

    assert request.config["retained"] == chr(0xD83D) + chr(0xDE00)


def test_unknown_schema_version_fails_closed() -> None:
    with pytest.raises(ReviewContractsValidationError, match="unknown schema version"):
        review_contracts.load_review_contracts_schema("review-bundle.v99")


def test_canonical_digest_ignores_location_but_binds_content() -> None:
    left = review_bundle_from_dict(_bundle_doc())
    relocated = _bundle_doc()
    relocated["episodes"][0]["references"][0]["uri"] = "elsewhere/trace-0000.json"  # type: ignore[index,union-attr]
    right = review_bundle_from_dict(relocated)
    assert review_bundle_canonical_digest(left) != review_bundle_canonical_digest(right)
    twin = review_bundle_from_dict(_bundle_doc())
    assert review_bundle_canonical_digest(left) == review_bundle_canonical_digest(twin)


def test_deterministic_rerun_matches_artifact_digest(tmp_path: Path) -> None:
    def once(directory: str) -> str:
        request = component_request_from_dict(
            {
                "schema_version": "component-request.v1",
                "request_id": "r",
                "component_id": "srev01-capability-report",
                "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
                "output_directory": directory,
            }
        )
        result = run(request, base=tmp_path)
        assert result.status == "complete"
        return str(result.artifacts[0]["sha256"])

    assert once("run-a") == once("run-b")


def test_admitted_source_resolves_checked_in_fixture() -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt_path = ADMITTED_SOURCE_FIXTURE_DIR / "receipt.json"

    result = resolve_admitted_source(
        receipt_path,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert result.status == "admitted"
    assert result.reason == "admitted"
    assert result.source_path == (ADMITTED_SOURCE_FIXTURE_DIR / "source.json").resolve()
    assert result.source_bytes == (ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes()
    assert (
        result.source_path.read_bytes()
        == (ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes()
    )
    assert result.receipt is not None
    assert result.receipt.source.schema == "fixture-source.v1"
    assert result.receipt.scientific_claim_allowed is False


def test_admitted_source_digest_helpers_match_fixture_receipt() -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = load_admitted_source_receipt(ADMITTED_SOURCE_FIXTURE_DIR / "receipt.json")

    assert component_request_canonical_digest(request) == receipt.request_sha256
    assert experiment_recipe_canonical_digest(recipe) == receipt.recipe_sha256


def test_component_request_digest_binds_config_but_ignores_output_directory() -> None:
    request = _admitted_source_fixture("request.json")
    relocated = copy.deepcopy(request)
    relocated["output_directory"] = "another-output"
    changed_config = copy.deepcopy(request)
    changed_config["config"] = {"changed": True}

    assert component_request_canonical_digest(relocated) == component_request_canonical_digest(
        request
    )
    assert component_request_canonical_digest(changed_config) != component_request_canonical_digest(
        request
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "changed-source.v2"),
        ("sha256", "c" * 64),
        ("source_commit", "c" * 40),
        ("config_identity", "changed-config.v2"),
        ("units", "feet"),
        ("coordinate_frame", "camera"),
    ],
)
def test_component_request_digest_binds_every_source_declaration(field: str, value: str) -> None:
    """Every v1 source declaration changes the canonical request identity when supplied."""
    request = _admitted_source_fixture("request.json")
    changed = copy.deepcopy(request)
    changed["sources"][0][field] = value

    assert component_request_canonical_digest(changed) != component_request_canonical_digest(
        request
    )


def test_component_request_digest_normalizes_omitted_optional_source_declarations() -> None:
    """The request parser gives every omitted v1 source declaration an empty-text value."""
    request = _admitted_source_fixture("request.json")
    request["sources"][0].pop("units")
    request["sources"][0].pop("coordinate_frame")
    parsed = component_request_from_dict(request)

    assert parsed.sources[0].schema == ""
    assert parsed.sources[0].sha256 == ""
    assert parsed.sources[0].source_commit == ""
    assert parsed.sources[0].config_identity == ""
    assert parsed.sources[0].units == ""
    assert parsed.sources[0].coordinate_frame == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "changed-source.v2"),
        ("sha256", "c" * 64),
        ("source_commit", "c" * 40),
        ("config_identity", "changed-config.v2"),
        ("units", "feet"),
        ("coordinate_frame", "camera"),
    ],
)
def test_admitted_source_rejects_request_source_declaration_mismatch(
    field: str, value: str
) -> None:
    """A supplied current request declaration must agree with the receipt, not only its digest."""
    request = _admitted_source_fixture("request.json")
    request["sources"][0][field] = value
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    receipt["request_sha256"] = component_request_canonical_digest(request)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert f"request source {field}" in result.detail
    assert result.source_path is None
    assert result.source_bytes is None


@pytest.mark.parametrize("field", ["units", "coordinate_frame"])
def test_admitted_source_rejects_missing_request_semantic_binding(field: str) -> None:
    """Omitting a semantic request declaration cannot silently widen admission."""
    request = _admitted_source_fixture("request.json")
    request["sources"][0].pop(field)
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    receipt["request_sha256"] = component_request_canonical_digest(request)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert f"request source {field}" in result.detail
    assert result.source_path is None


@pytest.mark.parametrize("field", ["units", "coordinate_frame"])
def test_admitted_source_rejects_recipe_semantic_mutation(field: str) -> None:
    """Recipe semantic declarations must match the receipt exactly."""
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    recipe["source_identity"][field] = "mutated-semantic-value"
    receipt = _admitted_source_fixture("receipt.json")
    receipt["recipe_sha256"] = experiment_recipe_canonical_digest(recipe)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert f"recipe source_identity.{field}" in result.detail
    assert result.source_path is None


@pytest.mark.parametrize("field", ["units", "coordinate_frame"])
def test_admitted_source_rejects_missing_recipe_semantic_binding(field: str) -> None:
    """Missing recipe semantic declarations cannot silently widen admission."""
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    recipe["source_identity"].pop(field)
    receipt = _admitted_source_fixture("receipt.json")
    receipt["recipe_sha256"] = experiment_recipe_canonical_digest(recipe)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert f"recipe source_identity.{field}" in result.detail
    assert result.source_path is None


def test_admitted_source_reports_missing_receipt_and_source(tmp_path: Path) -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt_path = tmp_path / "missing-receipt.json"

    missing_receipt = resolve_admitted_source(
        receipt_path,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )
    missing_source = resolve_admitted_source(
        ADMITTED_SOURCE_FIXTURE_DIR / "receipt.json",
        allowed_root=tmp_path,
        request=request,
        recipe=recipe,
    )

    assert (missing_receipt.status, missing_receipt.reason) == ("unavailable", "receipt_missing")
    assert (missing_source.status, missing_source.reason) == ("unavailable", "source_missing")
    assert missing_source.source_path is None


def test_admitted_source_bounds_oversized_json_integer_errors(tmp_path: Path) -> None:
    receipt_path = tmp_path / "oversized-receipt.json"
    receipt_path.write_text('{"oversized": ' + "9" * 5001 + "}", encoding="utf-8")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")

    with pytest.raises(ReviewContractsValidationError) as error:
        load_admitted_source_receipt(receipt_path)
    assert "invalid or exceeds parser limits" in str(error.value)
    assert len(str(error.value)) < 500

    result = resolve_admitted_source(
        receipt_path,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_unreadable")
    assert result.detail == "receipt JSON is invalid or exceeds parser limits"
    assert len(result.detail) < 300
    assert result.source_path is None


def test_admitted_source_bounds_schema_validation_detail() -> None:
    """Malformed receipt diagnostics stay bounded for oversized invalid fields."""
    receipt = _admitted_source_fixture("receipt.json")
    receipt["receipt_id"] = "!" * 1_000_000

    result = resolve_admitted_source(receipt, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("failed", "receipt_malformed")
    assert len(result.detail) <= 200
    assert result.source_path is None


@pytest.mark.parametrize("path", [None, 123])
def test_admitted_source_loader_uses_typed_errors_for_invalid_paths(path: object) -> None:
    """Invalid public loader path values use the contract error type."""
    with pytest.raises(ReviewContractsValidationError, match="cannot read admitted-source receipt"):
        load_admitted_source_receipt(path)  # type: ignore[arg-type]


def test_admitted_source_bounds_aggregate_identity_diagnostics() -> None:
    """Many invalid current source entries produce one bounded stale detail."""
    request = _admitted_source_fixture("request.json")
    request["sources"] = [{"artifact_id": str(index)} for index in range(1_000)]
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "request identity is unusable" in result.detail
    assert len(result.detail) <= review_contracts.MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
    assert result.source_path is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "benchmark"),
        ("evidence_boundary", "benchmark"),
        ("scientific_claim_allowed", True),
        ("dependent_family_status", "shared_family"),
    ],
)
def test_admitted_source_rejects_recipe_boundary_mutation(field: str, value: object) -> None:
    """Recipe identity cannot widen a receipt beyond the diagnostic-only boundary."""
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    recipe["source_identity"][field] = value
    receipt = _admitted_source_fixture("receipt.json")
    receipt["recipe_sha256"] = experiment_recipe_canonical_digest(recipe)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "diagnostic-only boundary" in result.detail
    assert result.source_path is None


def test_admitted_source_rejects_oversized_receipt_input(tmp_path: Path) -> None:
    """Receipt loading and resolution enforce a bounded serialized input size."""
    receipt_path = tmp_path / "oversized-receipt.json"
    receipt_path.write_bytes(b"{}" + b" " * review_contracts.MAX_ADMITTED_SOURCE_RECEIPT_BYTES)

    with pytest.raises(ReviewContractsValidationError, match="maximum size"):
        load_admitted_source_receipt(receipt_path)

    result = resolve_admitted_source(receipt_path, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("unavailable", "receipt_unreadable")
    assert "maximum size" in result.detail
    assert len(result.detail) <= review_contracts.MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
    assert result.source_path is None


def test_admitted_source_rejects_oversized_source_before_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced source is bounded before hashing and never becomes admission evidence."""
    source_path = tmp_path / "source.json"
    source_path.write_bytes(b"x" * 17)
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    monkeypatch.setattr(review_contracts, "MAX_ADMITTED_SOURCE_BYTES", 16)

    result = resolve_admitted_source(
        receipt,
        allowed_root=tmp_path,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "source_too_large")
    assert "maximum size" in result.detail
    assert result.source_path is None


def test_admitted_source_loader_bounds_large_invalid_schema_value(tmp_path: Path) -> None:
    """Public receipt-loader errors must remain bounded for hostile fields."""
    receipt = _admitted_source_fixture("receipt.json")
    receipt["source"]["sha256"] = "!" * 1_000_000
    receipt_path = tmp_path / "large-invalid-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReviewContractsValidationError) as error:
        load_admitted_source_receipt(receipt_path)

    assert len(str(error.value)) < 500
    assert len(error.value.errors[0]) < 300


def test_admitted_source_bounds_long_uri_filesystem_diagnostics() -> None:
    """Long local URIs must return bounded unavailable results, not OSError."""
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    long_uri = "x" * 4_096
    request["sources"][0]["uri"] = long_uri
    recipe["source_identity"]["source_uri"] = long_uri
    receipt["source"]["uri"] = long_uri
    receipt["request_sha256"] = component_request_canonical_digest(request)
    receipt["recipe_sha256"] = experiment_recipe_canonical_digest(recipe)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "source_missing")
    assert len(result.detail) <= 200
    assert result.source_path is None


@pytest.mark.parametrize("field", ["format", "schema", "config_identity"])
def test_admitted_source_rejects_unicode_surrogates(field: str) -> None:
    """Receipt source identity strings cannot be mutually admitted with surrogates."""
    receipt = _admitted_source_fixture("receipt.json")
    receipt["source"][field] = chr(0xD800)

    result = resolve_admitted_source(receipt, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("failed", "receipt_malformed")
    assert result.detail == "payload contains unsupported Unicode surrogate"
    assert result.source_path is None


def test_admitted_source_rejects_deeply_nested_receipt(tmp_path: Path) -> None:
    """Parser recursion limits become bounded loader and resolver rejections."""
    receipt_path = tmp_path / "deep-receipt.json"
    receipt_path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")

    with pytest.raises(ReviewContractsValidationError) as error:
        load_admitted_source_receipt(receipt_path)
    assert "invalid or exceeds parser limits" in str(error.value)
    assert len(str(error.value)) < 300

    result = resolve_admitted_source(receipt_path, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("unavailable", "receipt_unreadable")
    assert result.detail == "receipt JSON is invalid or exceeds parser limits"
    assert result.source_path is None


@pytest.mark.parametrize("source_kind", [[], {}])
def test_admitted_source_rejects_unhashable_source_kind(source_kind: object) -> None:
    """Untyped boundary values fail as malformed receipts before set membership."""
    receipt = _admitted_source_fixture("receipt.json")
    receipt["source_kind"] = source_kind

    result = resolve_admitted_source(receipt, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("failed", "receipt_malformed")
    assert result.detail == "source_kind must be a string"
    assert result.source_path is None


def test_admitted_source_rejects_unreprable_schema_version() -> None:
    """Malformed schema versions cannot leak integer repr conversion errors."""
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        huge_version = int("9" * 5_000)
    finally:
        sys.set_int_max_str_digits(previous_limit)

    receipt = _admitted_source_fixture("receipt.json")
    receipt["schema_version"] = huge_version

    result = resolve_admitted_source(receipt, allowed_root=ADMITTED_SOURCE_FIXTURE_DIR)

    assert (result.status, result.reason) == ("failed", "receipt_malformed")
    assert result.detail == "schema_version must be a string"
    assert len(result.detail) < 300
    assert result.source_path is None


@pytest.mark.parametrize("deep_field", ["request", "recipe"])
def test_admitted_source_rejects_deep_current_identity_mapping(deep_field: str) -> None:
    """Deep request or recipe identity data cannot escape digest binding."""
    receipt = _admitted_source_fixture("receipt.json")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    deep_value: object = []
    for _ in range(10_000):
        deep_value = [deep_value]
    if deep_field == "request":
        request["config"] = {"deep": deep_value}
    else:
        recipe["control_conditions"] = {"deep": deep_value}

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "identity is unusable" in result.detail
    assert len(result.detail) < 300
    assert result.source_path is None


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (
            "payload",
            ("failed", "receipt_malformed", "receipt payload resolver returned no payload"),
        ),
        ("parse", ("failed", "receipt_malformed", "receipt parser returned no receipt")),
        (
            "root",
            ("unavailable", "allowed_root_invalid", "allowed-root resolver returned no path"),
        ),
        (
            "source",
            ("unavailable", "source_missing", "source-path resolver returned no path"),
        ),
    ],
)
def test_admitted_source_fails_closed_when_resolver_returns_no_value(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected: tuple[str, str, str],
) -> None:
    """Impossible helper states return bounded rejection results without asserts."""
    receipt = _admitted_source_fixture("receipt.json")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    if stage == "payload":
        monkeypatch.setattr(review_contracts, "_receipt_payload", lambda _: (None, None))
    elif stage == "parse":
        monkeypatch.setattr(
            review_contracts,
            "_parse_admitted_source_receipt",
            lambda _: (None, None),
        )
    elif stage == "root":
        monkeypatch.setattr(review_contracts, "_resolve_allowed_root", lambda _: (None, None))
    else:
        monkeypatch.setattr(
            review_contracts,
            "_resolve_source_path",
            lambda _, __: (None, None),
        )

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason, result.detail) == expected
    assert result.source_path is None


def test_admitted_source_rejects_mutated_bytes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_bytes((ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes())
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\nmutation\n", encoding="utf-8"
    )
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")

    result = resolve_admitted_source(receipt, allowed_root=tmp_path, request=request, recipe=recipe)

    assert (result.status, result.reason) == ("failed", "source_mutated")
    assert result.source_path is None


def test_admitted_source_rejects_request_config_mutation() -> None:
    request = _admitted_source_fixture("request.json")
    stale_request = copy.deepcopy(request)
    stale_request["config"] = {"changed": True}
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    receipt["request_sha256"] = component_request_canonical_digest(request)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=stale_request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "request SHA-256" in result.detail
    assert result.source_path is None


@pytest.mark.parametrize("missing", ["request", "recipe", "both"])
def test_admitted_source_rejects_expected_values_without_current_context(missing: str) -> None:
    receipt = _admitted_source_fixture("receipt.json")
    request = None if missing in {"request", "both"} else _admitted_source_fixture("request.json")
    recipe = None if missing in {"recipe", "both"} else _admitted_source_fixture("recipe.json")

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
        expected_request_sha256=receipt["request_sha256"],
        expected_recipe_sha256=receipt["recipe_sha256"],
        expected_source_commit=receipt["source"]["source_commit"],
        expected_config_identity=receipt["source"]["config_identity"],
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "current request and recipe context" in result.detail
    assert result.source_path is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "admitted-source-receipt.v2"), ("source_kind", "benchmark")],
)
def test_admitted_source_rejects_unsupported_receipt_boundary(field: str, value: str) -> None:
    receipt = _admitted_source_fixture("receipt.json")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt[field] = value

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_unsupported")


def test_admitted_source_rejects_stale_request_and_recipe() -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt_path = ADMITTED_SOURCE_FIXTURE_DIR / "receipt.json"
    stale_request = copy.deepcopy(request)
    stale_request["request_id"] = "different-request"
    stale_recipe = copy.deepcopy(recipe)
    stale_recipe["hypothesis"] = "different-hypothesis"

    request_result = resolve_admitted_source(
        receipt_path,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=stale_request,
        recipe=recipe,
    )
    recipe_result = resolve_admitted_source(
        receipt_path,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=stale_recipe,
    )

    assert (request_result.status, request_result.reason) == ("unavailable", "receipt_stale")
    assert (recipe_result.status, recipe_result.reason) == ("unavailable", "receipt_stale")
    assert request_result.source_path is None
    assert recipe_result.source_path is None


def test_admitted_source_rejects_source_metadata_mismatch() -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")
    wrong_format = copy.deepcopy(receipt)
    wrong_format["source"]["format"] = "other-format"
    format_result = resolve_admitted_source(
        wrong_format,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )
    commit_result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
        expected_source_commit="c" * 40,
    )
    config_result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
        expected_config_identity="different-config",
    )

    assert (format_result.status, format_result.reason) == ("unavailable", "receipt_stale")
    assert (commit_result.status, commit_result.reason) == ("unavailable", "receipt_stale")
    assert (config_result.status, config_result.reason) == ("unavailable", "receipt_stale")


def test_admitted_source_rejects_escape_and_malformed_receipt(tmp_path: Path) -> None:
    receipt = _admitted_source_fixture("receipt.json")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_source = tmp_path / "outside-source.json"
    outside_source.write_bytes((ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes())
    try:
        (allowed_root / "source.json").symlink_to(outside_source)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")
    escaped_result = resolve_admitted_source(
        receipt,
        allowed_root=allowed_root,
        request=request,
        recipe=recipe,
    )

    malformed = copy.deepcopy(receipt)
    malformed["source"]["sha256"] = "bad"
    malformed_result = resolve_admitted_source(
        malformed,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (escaped_result.status, escaped_result.reason) == (
        "unavailable",
        "source_escaped_root",
    )
    assert (malformed_result.status, malformed_result.reason) == ("failed", "receipt_malformed")


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
    or os.open not in os.supports_dir_fd,
    reason="protected descriptor flags are unavailable",
)
def test_admitted_source_rejects_special_file_without_blocking(tmp_path: Path) -> None:
    """A FIFO is rejected after nonblocking no-follow open and never read as source bytes."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    os.mkfifo(allowed_root / "source.json")
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")

    result = resolve_admitted_source(
        receipt,
        allowed_root=allowed_root,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "source_not_regular")
    assert result.source_path is None
    assert result.source_bytes is None


@pytest.mark.skipif(
    not hasattr(os, "mkfifo")
    or not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_NONBLOCK")),
    reason="FIFO or protected descriptor flags are unavailable",
)
def test_admitted_source_receipt_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    """A receipt FIFO is rejected before a read can wait for a writer."""
    receipt_path = tmp_path / "receipt.json"
    os.mkfifo(receipt_path)
    command = (
        "import json, sys; "
        "from robot_sf.analysis_workbench.review_contracts import resolve_admitted_source; "
        "result = resolve_admitted_source(sys.argv[1], allowed_root=sys.argv[2]); "
        "print(json.dumps(result.to_dict(), sort_keys=True))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command, str(receipt_path), str(tmp_path)],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert (payload["status"], payload["reason"]) == ("unavailable", "receipt_unreadable")
    assert payload["detail"] == "admitted-source receipt must be a regular file"


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
    or os.open not in os.supports_dir_fd,
    reason="protected descriptor flags are unavailable",
)
def test_admitted_source_rejects_symlink_replacement_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source replaced by an outside symlink before protected open cannot be admitted."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source_path = allowed_root / "source.json"
    source_path.write_bytes((ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes())
    outside_source = tmp_path / "outside-source.json"
    outside_source.write_bytes(b"outside\n")
    real_open = review_contracts.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if kwargs.get("dir_fd") is not None and path == "source.json" and not swapped:
            source_path.unlink()
            source_path.symlink_to(outside_source)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(review_contracts.os, "open", racing_open)
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")

    result = resolve_admitted_source(
        receipt,
        allowed_root=allowed_root,
        request=request,
        recipe=recipe,
    )

    assert swapped is True
    assert (result.status, result.reason) == ("unavailable", "source_escaped_root")
    assert result.source_path is None
    assert result.source_bytes is None


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
    or os.open not in os.supports_dir_fd,
    reason="protected descriptor flags are unavailable",
)
def test_admitted_source_bytes_survive_path_replacement_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protected bytes remain tied to the opened file when its pathname is replaced."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source_path = allowed_root / "source.json"
    source_bytes = (ADMITTED_SOURCE_FIXTURE_DIR / "source.json").read_bytes()
    source_path.write_bytes(source_bytes)
    outside_source = tmp_path / "outside-source.json"
    outside_source.write_bytes(b"outside\n")
    real_open = review_contracts.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        opened = real_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is not None and path == "source.json" and not swapped:
            source_path.unlink()
            source_path.symlink_to(outside_source)
            swapped = True
        return opened

    monkeypatch.setattr(review_contracts.os, "open", racing_open)
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    receipt = _admitted_source_fixture("receipt.json")

    result = resolve_admitted_source(
        receipt,
        allowed_root=allowed_root,
        request=request,
        recipe=recipe,
    )

    assert swapped is True
    assert result.status == "admitted"
    assert result.source_bytes == source_bytes
    assert result.source_path == source_path
    assert result.source_path.is_symlink()


def test_admitted_source_requires_v1_recipe_admission_reference() -> None:
    request = _admitted_source_fixture("request.json")
    recipe = _admitted_source_fixture("recipe.json")
    recipe.pop("admission_reference")
    receipt = _admitted_source_fixture("receipt.json")
    receipt["recipe_sha256"] = experiment_recipe_canonical_digest(recipe)

    result = resolve_admitted_source(
        receipt,
        allowed_root=ADMITTED_SOURCE_FIXTURE_DIR,
        request=request,
        recipe=recipe,
    )

    assert (result.status, result.reason) == ("unavailable", "receipt_stale")
    assert "migrate" in result.detail


def test_cli_stdout_is_a_component_result_v1_envelope(tmp_path: Path, capsys: Any) -> None:
    """The standalone CLI emits the shared result schema on successful requests."""
    from robot_sf.analysis_workbench.review_contracts import main

    request_path = Path("tests/fixtures/scenario_review/review_contracts/request.json")
    config_path = Path("tests/fixtures/scenario_review/review_contracts/config.json")
    code = main(
        [
            "--input",
            str(request_path),
            "--config",
            str(config_path),
            "--output",
            "cli-output",
            "--base",
            str(tmp_path),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    parsed = component_result_from_dict(printed)
    assert code == 0
    assert printed["schema_version"] == "component-result.v1"
    assert parsed.status == "complete"


@pytest.mark.parametrize("parser_input", ["request", "config"])
def test_cli_parser_limit_emits_stable_failed_result(
    tmp_path: Path, capsys: Any, parser_input: str
) -> None:
    """Request and override config parser limits never leak a traceback."""
    from robot_sf.analysis_workbench.review_contracts import main

    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    huge_integer = "9" * 5001
    if parser_input == "request":
        request_path.write_text('{"config":{"seed":' + huge_integer + "}}", encoding="utf-8")
    else:
        request_path.write_text("{}", encoding="utf-8")
        config_path.write_text('{"seed":' + huge_integer + "}", encoding="utf-8")

    arguments = [
        "--input",
        str(request_path),
        "--output",
        "parser-limit-output",
        "--base",
        str(tmp_path),
    ]
    if parser_input == "config":
        arguments.extend(["--config", str(config_path)])

    exit_code = main(arguments)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["schema_version"] == "component-result.v1"
    result = component_result_from_dict(printed)
    assert result.status == "failed"
    assert result.reason == f"invalid_input: {parser_input} JSON cannot be parsed safely"
    assert not (tmp_path / "parser-limit-output").exists()


@pytest.mark.skipif(
    not hasattr(os, "mkfifo")
    or not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_NONBLOCK")),
    reason="FIFO or protected descriptor flags are unavailable",
)
@pytest.mark.parametrize("parser_input", ["request", "config"])
def test_cli_fifo_input_is_rejected_without_blocking(tmp_path: Path, parser_input: str) -> None:
    """CLI request and config FIFOs fail quickly without waiting for a writer."""
    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "fifo-request",
                "component_id": "srev01-inspect",
                "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
                "output_directory": "unused",
            }
        ),
        encoding="utf-8",
    )
    if parser_input == "request":
        request_path.unlink()
        os.mkfifo(request_path)
    else:
        os.mkfifo(config_path)

    arguments = [
        "--input",
        str(request_path),
        "--output",
        "fifo-output",
        "--base",
        str(tmp_path),
    ]
    if parser_input == "config":
        arguments.extend(["--config", str(config_path)])
    command = (
        "from robot_sf.analysis_workbench.review_contracts import main; "
        "raise SystemExit(main(__import__('sys').argv[1:]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command, *arguments],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    printed = json.loads(completed.stdout)
    assert printed["schema_version"] == "component-result.v1"
    assert printed["request_id"] == ("unknown" if parser_input == "request" else "fifo-request")
    assert printed["component_id"] == ("unknown" if parser_input == "request" else "srev01-inspect")
    assert printed["reason"] == f"invalid_input: {parser_input} JSON cannot be parsed safely"
    assert component_result_from_dict(printed).status == "failed"
    assert not (tmp_path / "fifo-output").exists()


def test_cli_failure_envelope_bounds_identities(tmp_path: Path, capsys: Any) -> None:
    """Malformed CLI identities are replaced with bounded safe envelope values."""
    from robot_sf.analysis_workbench.review_contracts import main

    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "r" * (review_contracts.MAX_REVIEW_CONTRACT_ID_CHARS + 1),
                "component_id": "c" * (review_contracts.MAX_REVIEW_CONTRACT_ID_CHARS + 1),
                "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
                "output_directory": "bounded-id-output",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "bounded-id-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["request_id"] == "unknown"
    assert printed["component_id"] == "unknown"
    assert len(printed["reason"]) <= review_contracts.MAX_REVIEW_CONTRACT_REASON_CHARS
    assert component_result_from_dict(printed).status == "failed"


def test_cli_failure_envelope_bounds_interpolated_details(tmp_path: Path, capsys: Any) -> None:
    """CLI-generated missing-capability details remain bounded before serialization."""
    from robot_sf.analysis_workbench.review_contracts import main

    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "bounded-detail-request",
                "component_id": "srev01-inspect",
                "sources": [{"artifact_id": "a", "uri": "a.json", "format": "f"}],
                "required_capabilities": [f"unsupported-{index}" for index in range(100)],
                "output_directory": "bounded-detail-output",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "bounded-detail-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["reason"].startswith("missing capabilities: ")
    assert len(printed["reason"]) <= review_contracts.MAX_REVIEW_CONTRACT_REASON_CHARS
    assert component_result_from_dict(printed).status == "unavailable"
    assert not (tmp_path / "bounded-detail-output").exists()


def test_cli_nonstandard_json_constant_emits_stable_failed_result(
    tmp_path: Path, capsys: Any
) -> None:
    """NaN must be rejected before it can bypass the result envelope."""
    from robot_sf.analysis_workbench.review_contracts import main

    request_path = tmp_path / "request.json"
    request_path.write_text(
        '{"schema_version":"component-request.v1",'
        '"request_id":"r","component_id":"srev01-inspect",'
        '"sources":[{"artifact_id":"a","uri":"a.json","format":"f"}],'
        '"config":{"seed":NaN},"output_directory":"unused"}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "nan-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["schema_version"] == "component-result.v1"
    assert printed["reason"] == "invalid_input: request JSON cannot be parsed safely"
    assert component_result_from_dict(printed).status == "failed"
    assert not (tmp_path / "nan-output").exists()
