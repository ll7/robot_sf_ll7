"""Focused tests for the SREV-25 recorded-results comparison component."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import review_experiment_report as report_module
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_experiment_report import (
    COMPONENT_ID,
    EXPERIMENT_RESULTS_SCHEMA_VERSION,
    component_descriptor,
    main,
    run,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureResult


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_RESULTS = REPO_ROOT / "tests/fixtures/scenario_review/review_experiment_report/results.json"


def _request(
    *,
    output_directory: str = "out",
    source_uri: str = "results.json",
    source_sha256: str = "",
    source_commit: str = "",
    required_capabilities: tuple[str, ...] = (),
    config: dict[str, Any] | None = None,
) -> ComponentRequest:
    return ComponentRequest(
        request_id="test-request",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="recorded-results",
                uri=source_uri,
                format=EXPERIMENT_RESULTS_SCHEMA_VERSION,
                sha256=source_sha256,
                source_commit=source_commit,
            ),
        ),
        output_directory=output_directory,
        config=config or {},
        required_capabilities=required_capabilities,
    )


def _copy_fixture(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_bytes(FIXTURE_RESULTS.read_bytes())


def _write_results(tmp_path: Path, payload: dict[str, Any]) -> None:
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_raw_results(tmp_path: Path, content: str) -> None:
    (tmp_path / "results.json").write_text(content, encoding="utf-8")


def _load_report(output_directory: Path) -> dict[str, Any]:
    return json.loads((output_directory / "experiment-comparison.json").read_text(encoding="utf-8"))


def test_descriptor_declares_recorded_results_and_versioned_outputs() -> None:
    descriptor = component_descriptor()

    assert descriptor["schema_version"] == COMPONENT_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor["component_id"] == COMPONENT_ID
    assert descriptor["supported_input_versions"] == ["component-request.v1"]
    assert descriptor["required_capabilities"] == ["recorded-experiment-results"]
    assert descriptor["output_types"] == [
        "experiment-comparison.v1",
        "experiment-comparison-html.v1",
    ]
    round_tripped = component_descriptor_from_dict(descriptor)
    assert round_tripped.component_id == COMPONENT_ID
    assert round_tripped.optional_capabilities == ("activation-status",)


def test_control_id_must_reference_the_role_control_condition(tmp_path: Path) -> None:
    """A treatment cannot be used as the control for an admitted comparison."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["control_condition_id"] = "delay-100ms"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "must identify the role=control condition" in result.reason
    assert not (tmp_path / "out").exists()


def test_missing_source_identity_fails_closed(tmp_path: Path) -> None:
    """A source without closed provenance cannot publish a verified report."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"] = {}
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "source_identity is missing required keys" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_unknown_source_identity_key_fails_closed(tmp_path: Path) -> None:
    """Unvalidated provenance extensions cannot be silently retained."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["unvalidated_note"] = "not part of the contract"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "source_identity has unsupported keys: unvalidated_note" in result.reason
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_kind", "forged-kind"),
        ("execution_mode", "unbounded-mode"),
        ("source_commit", "not-a-commit"),
    ],
)
def test_source_provenance_uses_bounded_values(tmp_path: Path, field: str, value: str) -> None:
    """A verified report requires semantically bounded source provenance."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"][field] = value
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert not (tmp_path / "out").exists()


def test_declared_source_ref_integrity_is_verified(tmp_path: Path) -> None:
    """Optional request-level source digests and commits bind the source bytes."""
    _copy_fixture(tmp_path)
    source_sha256 = hashlib.sha256((tmp_path / "results.json").read_bytes()).hexdigest()

    result = run(
        _request(source_sha256=source_sha256, source_commit="a" * 40),
        base=tmp_path,
    )

    assert result.status == "complete"
    assert _load_report(tmp_path / "out")["source"]["sha256"] == source_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_sha256", "0" * 64, "declared source sha256"),
        ("source_commit", "b" * 40, "declared source_commit"),
    ],
)
def test_declared_source_ref_mismatch_fails_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    """A substituted source cannot pass an explicit request-level integrity check."""
    _copy_fixture(tmp_path)
    request = _request(**{field: value})

    result = run(request, base=tmp_path)

    assert result.status == "failed"
    assert result.reason == f"source_integrity: {message} does not match " + (
        "observed source bytes" if field == "source_sha256" else "source identity"
    )
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("readiness_status", ["missing", "fallback", "degraded", "unavailable"])
def test_non_verified_source_status_is_preserved_and_taints_effects(
    tmp_path: Path, readiness_status: str
) -> None:
    """Diagnostic source statuses remain visible and cannot publish verified effects."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["readiness_status"] = readiness_status
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert len(result.artifacts) == 2
    report = _load_report(tmp_path / "out")
    assert report["comparison_status"] == "diagnostic_tainted"
    assert report["source"]["readiness_status"] == readiness_status
    assert report["provenance"]["source_readiness_status"] == readiness_status
    assert all(
        effect["reason"] == "source_readiness_not_verified"
        for family in report["families"]
        for effect in family["effects"]
    )


def test_non_available_source_status_is_preserved_and_taints_effects(tmp_path: Path) -> None:
    """An unavailable source cannot be treated as a verified comparison."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["availability_status"] = "not_available"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    report = _load_report(tmp_path / "out")
    assert report["comparison_status"] == "diagnostic_tainted"
    assert report["source"]["availability_status"] == "not_available"
    assert report["provenance"]["source_availability_status"] == "not_available"
    assert all(
        effect["reason"] == "source_availability_not_verified"
        for family in report["families"]
        for effect in family["effects"]
    )


def test_expected_direction_rejects_unknown_enum_member(tmp_path: Path) -> None:
    """Authoritative measurements cannot retain an unsupported direction."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["conditions"][1]["measurements"]["clearance_m"]["expected_direction"] = (
        "sideways"
    )
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "expected_direction must be one of increase, decrease" in result.reason
    assert not (tmp_path / "out").exists()


def test_valid_expected_direction_is_retained(tmp_path: Path) -> None:
    """Supported expected-direction values remain available to report consumers."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["conditions"][1]["measurements"]["clearance_m"]["expected_direction"] = (
        "decrease"
    )
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    report = _load_report(tmp_path / "out")
    treatment = report["families"][0]["treatments"][0]["condition"]
    assert treatment["measurements"]["clearance_m"]["expected_direction"] == "decrease"


def test_unknown_config_key_surrogate_is_not_retained_in_direct_api_error(
    tmp_path: Path,
) -> None:
    """Direct API errors remain UTF-8-safe even for malformed config keys."""
    _copy_fixture(tmp_path)

    result = run(_request(config={"\ud800": "value"}), base=tmp_path)

    assert result.status == "failed"
    assert "Unicode surrogate" in result.reason
    result.reason.encode("utf-8")
    assert not (tmp_path / "out").exists()


def test_run_keeps_dependent_families_and_negative_outcomes(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    result = run(
        _request(config={"metric_order": ["clearance_m", "success_rate"]}),
        base=tmp_path,
    )

    assert result.status == "complete"
    assert len(result.artifacts) == 2
    report = _load_report(tmp_path / "out")
    assert report["summary"] == {
        "condition_count": 5,
        "count_units": {
            "condition_count": "condition_record_including_control_and_treatment",
            "effect_counts": "treatment_metric_comparison_record",
            "family_count": "dependent_family_by_shared_parent_id",
            "negative_finding_count": "condition_record_with_non_survived_outcome",
            "outcome_counts": "condition_record_including_control_and_treatment",
        },
        "effect_counts": {"blocked": 4, "interpretable": 2},
        "executor_required": False,
        "family_count": 2,
        "independence": {
            "count_published": False,
            "reason": "no independence contract was supplied or validated",
            "status": "not_validated",
        },
        "negative_finding_count": 3,
        "outcome_counts": {
            "contradictory": 1,
            "falsified": 1,
            "inconclusive": 1,
            "survived": 2,
        },
        "recorded_results_only": True,
        "sample_unit": "dependent_family_by_shared_parent_id",
        "source_availability_status": "available",
        "source_readiness_status": "verified",
    }
    assert "independent_family_count" not in report["summary"]
    assert report["comparison_status"] == "verified"
    first_family = report["families"][0]
    assert first_family["sample_unit"] == "dependent_family_by_shared_parent_id"
    assert first_family["shared_parent_id"] == "episode-parent-0001"
    assert first_family["effect_interpretation"] == "allowed"
    clearance_effect = first_family["effects"][0]
    assert clearance_effect == {
        "control_value": 1.2,
        "delta": -0.5,
        "direction": "decrease",
        "metric": "clearance_m",
        "reason": None,
        "status": "interpretable",
        "treatment_value": 0.7,
        "units": "m",
    }
    assert {item["outcome"] for item in report["negative_findings"]} == {
        "contradictory",
        "falsified",
        "inconclusive",
    }
    assert report["summary"]["executor_required"] is False
    assert report["source"]["readiness_status"] == "verified"
    assert report["source"]["availability_status"] == "available"
    assert report["provenance"]["source_execution_mode"] == "recorded_results_only"
    assert "not benchmark" in report["evidence_boundary"]
    html_report = (tmp_path / "out" / "experiment-comparison.html").read_text(encoding="utf-8")
    assert "Outcome inventory" in html_report
    assert "Negative findings" in html_report
    assert "100 ms delay" in html_report
    assert "diagnostic-only" in html_report
    assert "JSON artifact is authoritative" in html_report
    assert "source provenance" in html_report

    result_document = json.loads(
        json.dumps({"schema_version": "component-result.v1", **asdict(result)})
    )
    component_result_from_dict(result_document)
    for artifact in result.artifacts:
        local_path = tmp_path / str(artifact["uri"])
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == artifact["sha256"]


def test_failed_control_blocks_all_effects_but_retains_contradictory_outcome(
    tmp_path: Path,
) -> None:
    _copy_fixture(tmp_path)
    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    report = _load_report(tmp_path / "out")
    failed_control_family = report["families"][1]
    assert failed_control_family["effect_interpretation"] == "blocked"
    assert all(effect["status"] == "blocked" for effect in failed_control_family["effects"])
    assert all(
        effect["reason"] == "control_fidelity_not_verified"
        for effect in failed_control_family["effects"]
    )
    contradictory = failed_control_family["treatments"][0]["condition"]
    assert contradictory["outcome"] == "contradictory"
    assert contradictory["activation"]["status"] == "pass"


def test_empty_treatment_measurements_retain_outcome_and_block_only_its_effects(
    tmp_path: Path,
) -> None:
    """An empty treatment measurement set blocks its effects but retains the outcome."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["conditions"][1]["measurements"] = {}
    payload["families"][0]["conditions"][2]["activation"]["status"] = "pass"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    first_family = _load_report(tmp_path / "out")["families"][0]
    empty_treatment = first_family["treatments"][0]
    assert empty_treatment["condition"]["outcome"] == "falsified"
    assert empty_treatment["condition"]["measurements"] == {}
    assert all(effect["status"] == "blocked" for effect in empty_treatment["effects"])
    assert all(
        effect["reason"] == "measurement_missing_in_one_condition"
        for effect in empty_treatment["effects"]
    )
    remaining_treatment_effects = {
        effect["metric"]: effect for effect in first_family["treatments"][1]["effects"]
    }
    assert remaining_treatment_effects["success_rate"]["status"] == "interpretable"
    assert remaining_treatment_effects["clearance_m"]["reason"] == "measurement_value_missing"


def test_oversized_integer_measurement_fails_closed(tmp_path: Path) -> None:
    """An integer that cannot become a finite float returns a stable input failure."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["conditions"][1]["measurements"]["clearance_m"]["value"] = 10**400
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "/value" in result.reason
    assert not (tmp_path / "out").exists()


def test_oversized_json_integer_fails_closed_without_exception(tmp_path: Path) -> None:
    """The JSON parser's digit limit becomes a bounded invalid-input result."""
    source = FIXTURE_RESULTS.read_text(encoding="utf-8")
    needle = '"value": 0.7'
    replacement = f'"value": {"9" * 5001}'
    assert needle in source
    _write_raw_results(tmp_path, source.replace(needle, replacement, 1))

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert len(result.reason) < 500
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_deeply_nested_source_fails_closed_without_exception(tmp_path: Path) -> None:
    """Recursive source validation returns a bounded failed result."""
    _write_raw_results(tmp_path, "[" * 1200 + "0" + "]" * 1200)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == "invalid_input: source JSON exceeds the supported nesting depth"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_finite_measurements_with_nonfinite_difference_fail_closed(tmp_path: Path) -> None:
    """Two finite values cannot publish an infinite treatment-control effect."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["families"][0]["conditions"][0]["measurements"]["clearance_m"]["value"] = -1.6e308
    payload["families"][0]["conditions"][1]["measurements"]["clearance_m"]["value"] = 1.6e308
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "difference for clearance_m must be finite" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_second_artifact_failure_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed HTML write must not publish the already-written JSON artifact."""
    _copy_fixture(tmp_path)
    original_write = report_module._write_artifact
    write_count = 0

    def fail_on_second_write(path: Path, content: str) -> str:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("synthetic second-artifact failure")
        return original_write(path, content)

    monkeypatch.setattr(report_module, "_write_artifact", fail_on_second_write)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("output_write_error:")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_lone_unicode_surrogate_fails_closed_without_partial_output(tmp_path: Path) -> None:
    """An escaped lone surrogate is rejected before either artifact is staged."""
    source = FIXTURE_RESULTS.read_text(encoding="utf-8")
    needle = '"hypothesis": "A delayed response changes clearance and task success."'
    replacement = r'"hypothesis": "A delayed response changes clearance and task success. \ud800"'
    assert needle in source
    _write_raw_results(tmp_path, source.replace(needle, replacement, 1))

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "Unicode" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_lone_unicode_surrogate_in_source_identity_fails_before_json_publish(
    tmp_path: Path,
) -> None:
    """A surrogate in JSON-only identity data cannot produce an authoritative artifact."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["generator"] = "\ud800"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "Unicode surrogate" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_lone_unicode_surrogate_in_source_key_fails_closed(tmp_path: Path) -> None:
    """Mapping keys receive the same strict-Unicode validation as string values."""
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["\ud800"] = "generator"
    _write_results(tmp_path, payload)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "Unicode surrogate" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_valid_escaped_surrogate_pair_remains_supported(tmp_path: Path) -> None:
    """A valid JSON UTF-16 pair still decodes to the same non-BMP character."""
    source = FIXTURE_RESULTS.read_text(encoding="utf-8")
    needle = '"source_kind": "fixture",'
    replacement = '"source_kind": "fixture",\n    "generator": "\\ud83e\\udd16",'
    assert needle in source
    _write_raw_results(tmp_path, source.replace(needle, replacement, 1))

    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    assert len(result.artifacts) == 2
    assert _load_report(tmp_path / "out")["source"]["identity"]["generator"] == "🤖"


def test_repeated_fixture_runs_have_identical_logical_artifact_digests(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    first = run(_request(output_directory="run-a"), base=tmp_path)
    second = run(_request(output_directory="run-b"), base=tmp_path)

    assert first.status == second.status == "complete"
    assert [item["sha256"] for item in first.artifacts] == [
        item["sha256"] for item in second.artifacts
    ]
    assert _load_report(tmp_path / "run-a") == _load_report(tmp_path / "run-b")


def test_corrupt_source_fails_without_partial_output(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text("{not json", encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.artifacts == ()
    assert result.reason.startswith("invalid_input:")
    assert not (tmp_path / "out").exists()


def test_duplicate_source_json_object_names_fail_closed(tmp_path: Path) -> None:
    """Duplicate source members cannot be resolved by parser order."""
    source = FIXTURE_RESULTS.read_text(encoding="utf-8")
    needle = '"readiness_status": "verified",'
    replacement = needle + '\n    "readiness_status": "missing",'
    assert needle in source
    _write_raw_results(tmp_path, source.replace(needle, replacement, 1))

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == "invalid_input: source JSON contains duplicate object names"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_oversized_source_fails_before_reading_contents(tmp_path: Path) -> None:
    """The source byte cap is enforced before parsing or report publication."""
    source_path = tmp_path / "results.json"
    source_path.write_bytes(b" " * (report_module.MAX_SOURCE_BYTES + 1))

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == (
        f"source_too_large: source exceeds the {report_module.MAX_SOURCE_BYTES}-byte limit"
    )
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_non_regular_source_is_rejected_before_read(tmp_path: Path) -> None:
    """Special files receive a bounded refusal instead of a potentially blocking read."""
    os.mkfifo(tmp_path / "results.fifo")

    result = run(_request(source_uri="results.fifo"), base=tmp_path)

    assert result.status == "unavailable"
    assert result.reason == "source_unavailable: source path must be a regular file"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_missing_capability_is_unavailable_without_output(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)

    result = run(
        _request(required_capabilities=("video-frames",)),
        base=tmp_path,
    )

    assert result.status == "unavailable"
    assert result.reason == "missing_capability: video-frames"
    assert not (tmp_path / "out").exists()


def test_incompatible_source_version_is_unavailable(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["schema_version"] = "experiment-results.v99"
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "unavailable"
    assert result.reason.startswith("incompatible_version:")
    assert not (tmp_path / "out").exists()


def test_nonfinite_source_extension_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_RESULTS.read_text(encoding="utf-8"))
    payload["source_identity"]["nonfinite"] = float("nan")
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "non-finite" in result.reason
    assert not (tmp_path / "out").exists()


def test_output_collision_and_invalid_config_fail_closed(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    (tmp_path / "out").mkdir()
    collision = run(_request(), base=tmp_path)
    invalid_config = run(
        _request(output_directory="other", config={"arbitrary_code": "ignored"}),
        base=tmp_path,
    )

    assert collision.status == "failed"
    assert collision.reason.startswith("output_collision:")
    assert invalid_config.status == "failed"
    assert invalid_config.reason.startswith("invalid_config:")
    assert not (tmp_path / "other").exists()


def test_concurrent_empty_output_directory_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent empty directory remains untouched at the final claim."""
    _copy_fixture(tmp_path)
    output_dir = tmp_path / "out"
    original_reserve = report_module._reserve_output_directory

    def create_before_reservation(path: Path) -> None:
        output_dir.mkdir()
        original_reserve(path)

    monkeypatch.setattr(report_module, "_reserve_output_directory", create_before_reservation)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("output_collision:")
    assert output_dir.is_dir()
    assert tuple(output_dir.iterdir()) == ()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_lone_unicode_surrogate_in_metric_order_fails_before_json_publish(
    tmp_path: Path,
) -> None:
    """Retained config strings use the same strict Unicode boundary as sources."""
    _copy_fixture(tmp_path)

    result = run(_request(config={"metric_order": ["\ud800"]}), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    assert "Unicode surrogate" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize("unsafe_text", ["title\x00with-control", "title\x01with-control"])
def test_control_text_in_report_title_fails_before_html_publish(
    tmp_path: Path, unsafe_text: str
) -> None:
    """NUL and other C0 controls cannot reach the HTML-safe report fields."""
    _copy_fixture(tmp_path)

    result = run(_request(config={"report_title": unsafe_text}), base=tmp_path)

    assert result.status == "failed"
    assert result.reason.startswith("invalid_input: /config/report_title")
    assert "control" in result.reason or "NUL" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("source_uri", "expected_reason"),
    [("bad\x00name", "invalid_input: /sources/0/uri contains an embedded NUL byte")],
)
def test_invalid_source_uri_returns_failed_result_without_output(
    tmp_path: Path, source_uri: str, expected_reason: str
) -> None:
    """Direct callers receive a bounded result for unsafe source path text."""
    request = _request(source_uri=source_uri)

    result = run(request, base=tmp_path)

    assert result.status == "failed"
    assert result.reason == expected_reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_surrogate_source_uri_is_rejected_at_request_boundary() -> None:
    """Lone surrogates are rejected before a component request is constructed."""
    with pytest.raises(ReviewContractsValidationError, match="Unicode surrogate"):
        component_request_from_dict(
            {
                "schema_version": "component-request.v1",
                "request_id": "invalid-uri-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": "bad\ud800",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "output_directory": "out",
            }
        )


def test_cli_reads_request_and_config_and_emits_result(
    tmp_path: Path, capsys: CaptureResult[str]
) -> None:
    _copy_fixture(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "cli-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": "results.json",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"report_title": "CLI fixture report"}),
        encoding="utf-8",
    )

    exit_code = main(
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
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 0
    assert printed["schema_version"] == COMPONENT_RESULT_SCHEMA_VERSION
    component_result_from_dict(printed)
    assert printed["status"] == "complete"
    assert (tmp_path / "cli-output" / "experiment-comparison.json").exists()
    assert _load_report(tmp_path / "cli-output")["title"] == "CLI fixture report"


def test_cli_deeply_nested_source_emits_failed_result(
    tmp_path: Path, capsys: CaptureResult[str]
) -> None:
    """The CLI converts recursive source validation failures into the result envelope."""
    _write_raw_results(tmp_path, "[" * 1200 + "0" + "]" * 1200)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "nested-cli-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": "results.json",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "cli-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert printed["schema_version"] == COMPONENT_RESULT_SCHEMA_VERSION
    result = component_result_from_dict(printed)
    assert result.status == "failed"
    assert "nesting depth" in result.reason
    assert not (tmp_path / "cli-output").exists()


@pytest.mark.parametrize("parser_input", ["request", "config"])
def test_cli_parser_limit_emits_stable_failed_result(
    tmp_path: Path, capsys: CaptureResult[str], parser_input: str
) -> None:
    """Request and override config parser limits never leak a traceback."""
    _copy_fixture(tmp_path)
    request = {
        "schema_version": "component-request.v1",
        "request_id": "parser-limit-request",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "recorded-results",
                "uri": "results.json",
                "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
            }
        ],
        "config": {},
        "output_directory": "ignored-by-cli",
    }
    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    huge_integer = "9" * 5001
    if parser_input == "request":
        request_path.write_text(
            json.dumps(request)[:-1] + f', "config": {{"metric_order": [{huge_integer}]}}}}',
            encoding="utf-8",
        )
    else:
        request_path.write_text(json.dumps(request), encoding="utf-8")
        config_path.write_text(f'{{"metric_order": [{huge_integer}]}}', encoding="utf-8")

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
    assert printed["schema_version"] == COMPONENT_RESULT_SCHEMA_VERSION
    result = component_result_from_dict(printed)
    assert result.status == "failed"
    assert result.reason == f"invalid_input: {parser_input} JSON cannot be parsed safely"
    assert not (tmp_path / "parser-limit-output").exists()


def test_cli_duplicate_config_object_names_emit_failed_result(
    tmp_path: Path, capsys: CaptureResult[str]
) -> None:
    """The optional config file uses the same duplicate-name rejection as sources."""
    _copy_fixture(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "duplicate-config-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": "results.json",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"report_title": "first", "report_title": "shadowed"}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--config",
            str(config_path),
            "--output",
            "duplicate-config-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    result = component_result_from_dict(json.loads(captured.out))

    assert exit_code == 1
    assert captured.err == ""
    assert result.request_id == "duplicate-config-request"
    assert result.status == "failed"
    assert result.reason == "invalid_input: config JSON contains duplicate object names"
    assert not (tmp_path / "duplicate-config-output").exists()


@pytest.mark.parametrize("source_uri", ["bad\x00name", "bad\ud800"])
def test_cli_invalid_source_uri_emits_failed_component_result(
    tmp_path: Path, capsys: CaptureResult[str], source_uri: str
) -> None:
    """Unsafe source URI text still produces the versioned CLI result envelope."""
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "invalid-uri-cli-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": source_uri,
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "cli-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["schema_version"] == COMPONENT_RESULT_SCHEMA_VERSION
    result = component_result_from_dict(printed)
    assert result.request_id == "invalid-uri-cli-request"
    assert result.component_id == COMPONENT_ID
    assert result.status == "failed"
    assert result.reason.startswith("invalid_input:")
    if source_uri == "bad\x00name":
        assert "/sources/0/uri" in result.reason
    else:
        assert result.reason == "invalid_input: request does not satisfy component-request.v1"
    assert result.artifacts == ()
    assert not (tmp_path / "cli-output").exists()


@pytest.mark.parametrize(
    ("artifact_id", "expected_reason"),
    [
        ("bad\x00id", "invalid_input: request does not satisfy component-request.v1"),
        ("bad\ud800", "invalid_input: request does not satisfy component-request.v1"),
    ],
)
def test_cli_invalid_source_identity_emits_failed_component_result(
    tmp_path: Path,
    capsys: CaptureResult[str],
    artifact_id: str,
    expected_reason: str,
) -> None:
    """Unsafe source identity text cannot be retained in a successful report."""
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "invalid-source-identity-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": artifact_id,
                        "uri": "results.json",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            "cli-output",
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    result = component_result_from_dict(json.loads(captured.out))

    assert exit_code == 1
    assert captured.err == ""
    assert result.request_id == "invalid-source-identity-request"
    assert result.component_id == COMPONENT_ID
    assert result.status == "failed"
    assert result.reason == expected_reason
    assert result.artifacts == ()
    assert not (tmp_path / "cli-output").exists()


@pytest.mark.parametrize("output_directory", ["bad\x00output", "bad\ud800output"])
def test_cli_invalid_output_path_emits_failed_component_result(
    tmp_path: Path, capsys: CaptureResult[str], output_directory: str
) -> None:
    """Unsafe CLI output paths are translated into failed result envelopes."""
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "component-request.v1",
                "request_id": "invalid-output-path-request",
                "component_id": COMPONENT_ID,
                "sources": [
                    {
                        "artifact_id": "recorded-results",
                        "uri": "results.json",
                        "format": EXPERIMENT_RESULTS_SCHEMA_VERSION,
                    }
                ],
                "config": {},
                "output_directory": "ignored-by-cli",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(request_path),
            "--output",
            output_directory,
            "--base",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    result = component_result_from_dict(json.loads(captured.out))

    assert exit_code == 1
    assert captured.err == ""
    assert result.request_id == "invalid-output-path-request"
    assert result.component_id == COMPONENT_ID
    assert result.status == "failed"
    if output_directory == "bad\x00output":
        assert result.reason.startswith("invalid_input: /output_directory")
    else:
        assert result.reason == "invalid_input: request does not satisfy component-request.v1"
    assert result.artifacts == ()


def test_path_resolution_errors_emit_failed_component_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected resolver failures stay inside the component result boundary."""
    original_resolve = Path.resolve

    def fail_for_source(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path.name == "results.json":
            raise ValueError("synthetic unsafe path error")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_for_source)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert result.request_id == "test-request"
    assert result.component_id == COMPONENT_ID
    assert result.reason == "invalid_input: /sources/0/uri cannot be resolved safely"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_run_accepts_relocated_output_without_changing_report_content(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    request = _request(output_directory="run-a")
    first = run(request, base=tmp_path)
    second = run(replace(request, output_directory="run-b"), base=tmp_path)

    assert first.status == second.status == "complete"
    assert _load_report(tmp_path / "run-a") == _load_report(tmp_path / "run-b")
