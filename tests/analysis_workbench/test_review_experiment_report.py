"""Focused tests for the SREV-25 recorded-results comparison component."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import review_experiment_report as report_module
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    ComponentRequest,
    SourceRef,
    component_descriptor_from_dict,
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
        "effect_counts": {"blocked": 4, "interpretable": 2},
        "executor_required": False,
        "family_count": 2,
        "independent_family_count": 2,
        "negative_finding_count": 3,
        "outcome_counts": {
            "contradictory": 1,
            "falsified": 1,
            "inconclusive": 1,
            "survived": 2,
        },
        "recorded_results_only": True,
        "sample_unit": "shared_parent_family",
    }
    first_family = report["families"][0]
    assert first_family["sample_unit"] == "dependent_family"
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
    assert printed["status"] == "complete"
    assert (tmp_path / "cli-output" / "experiment-comparison.json").exists()
    assert _load_report(tmp_path / "cli-output")["title"] == "CLI fixture report"


def test_run_accepts_relocated_output_without_changing_report_content(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    request = _request(output_directory="run-a")
    first = run(request, base=tmp_path)
    second = run(replace(request, output_directory="run-b"), base=tmp_path)

    assert first.status == second.status == "complete"
    assert _load_report(tmp_path / "run-a") == _load_report(tmp_path / "run-b")
