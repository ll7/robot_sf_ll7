"""Focused contract tests for SREV-01 review contracts (issue #9270)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench import review_contracts
from robot_sf.analysis_workbench.review_contracts import (
    ReviewContractsValidationError,
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
