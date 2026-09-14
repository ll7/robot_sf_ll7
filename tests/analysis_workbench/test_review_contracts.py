"""Focused contract tests for SREV-01 review contracts (issue #9270)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.analysis_workbench import review_contracts
from robot_sf.analysis_workbench.review_contracts import (
    ReviewContractsValidationError,
    component_request_from_dict,
    component_result_from_dict,
    experiment_recipe_from_dict,
    review_bundle_canonical_digest,
    review_bundle_from_dict,
    run,
    visualization_spec_from_dict,
)

COMMIT = "a" * 40
SHA = "b" * 64


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
