"""Tests for the issue #7340 real-manifest diagnostic contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import feasibility_first_real as real
from robot_sf.adversarial.attribution import FailureAttribution
from robot_sf.adversarial.bundle import write_candidate_inputs
from robot_sf.adversarial.certification_types import CertificationStatus
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    FeasibilityCandidate,
    FeasibilityCheck,
    HierarchicalScenarioValue,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "configs/benchmarks/issue_7340_feasibility_first_real_manifest_v1.yaml"
_SCENARIO_TEMPLATE = _REPO_ROOT / "configs/adversarial/issue_7340_station_platform_medium_v1.yaml"


def _candidate(seed: int = 301) -> CandidateSpec:
    """Build a compact candidate for helper-level checks."""
    return CandidateSpec(
        start=Pose2D(7.0, 21.5),
        goal=Pose2D(20.0, 21.5),
        spawn_time_s=0.5,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.25,
        scenario_seed=seed,
    )


def _route(*, eligible: str = "eligible", valid: bool = True) -> dict[str, Any]:
    """Build one compact scenario-cert route certificate."""
    return {
        "route_id": "route-1",
        "classification": "hard_but_solvable" if eligible == "eligible" else "knife_edge",
        "benchmark_eligibility": eligible,
        "checks": {
            "kinodynamic": {"command_limits_valid": valid},
            "inflated_collision_free_path": valid,
            "simulator_obstacle_collision": {"validated": valid, "collides_obstacle": False},
            "dynamic": {},
            "path_length_ratio": 1.4,
            "minimum_static_clearance_m": 1.0,
            "planned_turn_count": 2,
        },
    }


def _certification(
    *,
    status: str = "passed",
    routes: list[dict[str, Any]] | None = None,
    reason: str = "certified",
) -> CertificationStatus:
    """Build one scenario-cert status for helper and pipeline tests."""
    return CertificationStatus(
        schema_version="scenario_cert.v1",
        status=status,
        reason=reason,
        details={"certificates": [{"route_certificates": routes or [_route()]}]},
    )


def _checks() -> tuple[FeasibilityCheck, ...]:
    """Build four passing checks for method-summary tests."""
    return tuple(
        FeasibilityCheck(name, "pass", "evidence is available", {"source": "test"})
        for name in CHECK_NAMES
    )


def _feasible_candidate(candidate_id: str, *, family: str = "family") -> FeasibilityCandidate:
    """Build a candidate with deterministic typed feasibility evidence."""
    return FeasibilityCandidate(
        candidate_id=candidate_id,
        scenario_family=family,
        scenario_seed=301,
        control_hash="a" * 16,
        checks=_checks(),
        value=HierarchicalScenarioValue(0.5, 0.25, 0.75),
        feature_vector=(0.5, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0),
        candidate_controls={"test": True},
    )


def _fake_evaluator(
    config: Any,
    candidate: CandidateSpec,
    index: int,
) -> CandidateEvaluation:
    """Materialize a loader-valid bundle without executing the simulator."""
    candidate_dir = config.output_dir / f"candidate_{index:04d}"
    scenario_path, _route_path = write_candidate_inputs(
        config=config,
        candidate=candidate,
        candidate_dir=candidate_dir,
        index=index,
    )
    episode_path = candidate_dir / "episode_records.jsonl"
    episode_path.write_text(
        json.dumps({"status": "collision", "termination_reason": "collision"}) + "\n",
        encoding="utf-8",
    )
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=_certification(),
        objective_value=1.0,
        failure_attribution=FailureAttribution(
            status="attributed",
            primary_failure="collision",
            reasons=["test episode"],
            details={
                "execution_mode": "native",
                "availability_status": "available",
                "readiness_status": "native",
            },
        ),
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=scenario_path,
        bundle_path=candidate_dir,
    )


def test_real_manifest_pipeline_writes_schema_valid_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The config-first pipeline should preserve native rows and validate its report."""
    monkeypatch.setattr(real, "production_candidate_evaluator", lambda: _fake_evaluator)

    report = real.run_real_manifest_diagnostic(
        _MANIFEST,
        output_path=tmp_path / "report.json",
        output_dir=tmp_path / "bundles",
    )

    assert report["feasibility"]["feasible_candidates"] == 4
    assert report["governance"]["simulator_executed"] is True
    assert report["comparison"]["safety_event_severity"]["counts"] == {"collision": 4}
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["schema_version"] == (
        real.SCHEMA_VERSION
    )

    invalid = copy.deepcopy(report)
    invalid.pop("config")
    with pytest.raises(real.RealManifestError, match="config"):
        real.validate_real_report(invalid)


def test_real_manifest_pipeline_preserves_pipeline_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A production-pipeline exception remains an unavailable, schema-valid row."""

    def _raise(*_args: Any, **_kwargs: Any) -> CandidateEvaluation:
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(real, "production_candidate_evaluator", lambda: _raise)

    report = real.run_real_manifest_diagnostic(_MANIFEST, output_dir=tmp_path / "bundles")

    assert report["feasibility"]["unavailable_candidates"] == 4
    assert report["governance"]["simulator_executed"] is False
    assert report["comparison"]["methods"]["risk_feedback_hierarchical_value"]["status"] == (
        "unavailable"
    )


def test_hash_and_path_helpers_are_stable(tmp_path: Path) -> None:
    """Digest and input-resolution helpers preserve deterministic provenance."""
    input_path = tmp_path / "input.txt"
    input_path.write_bytes(b"fixture")

    assert real._sha256_bytes(b"fixture") == real._sha256_file(input_path)
    assert real._canonical_sha256({"b": 2, "a": 1}) == real._canonical_sha256({"a": 1, "b": 2})
    assert (
        real._repo_relative(_MANIFEST)
        == "configs/benchmarks/issue_7340_feasibility_first_real_manifest_v1.yaml"
    )
    assert (
        real._resolve_input_path(str(_MANIFEST), config_path=tmp_path / "config.yaml", field="x")
        == _MANIFEST
    )
    with pytest.raises(real.RealManifestError, match="x"):
        real._resolve_input_path("missing.yaml", config_path=tmp_path / "config.yaml", field="x")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unknown": True}), "unknown fields"),
        (lambda payload: payload.pop("policy"), "missing fields"),
        (lambda payload: payload.update({"baseline": "other"}), "baseline"),
        (
            lambda payload: payload["domain_approval"].update({"status": "approved"}),
            "domain_approval",
        ),
        (lambda payload: payload.update({"candidate_pool_budget": 0}), "candidate_pool_budget"),
        (lambda payload: payload.update({"sample_budget": 5}), "sample_budget"),
        (lambda payload: payload.update({"criticality_threshold": 2.0}), "criticality_threshold"),
        (lambda payload: payload.update({"dt": 0.0}), "dt"),
        (lambda payload: payload.update({"require_certification": False}), "require_certification"),
    ],
)
def test_real_manifest_config_fails_closed(mutation: Any, message: str, tmp_path: Path) -> None:
    """Manifest settings that weaken reproducibility or governance fail closed."""
    source = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    mutation(source)
    config_path = tmp_path / "manifest.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    with pytest.raises(real.RealManifestError, match=message):
        real._load_config(config_path)


def test_certificate_checks_cover_unavailable_failed_and_route_failures() -> None:
    """Certificate mapping keeps unavailable, failed, and ineligible routes distinct."""
    unavailable = real._certificate_checks(_certification(status="not_available"))
    assert [check.status for check in unavailable] == ["unavailable", "unavailable"]

    failed = real._certificate_checks(_certification(status="failed", reason="bad input"))
    assert [check.status for check in failed] == ["fail", "fail"]

    route_failed = real._certificate_checks(_certification(routes=[_route(valid=False)]))
    assert [check.status for check in route_failed] == ["fail", "fail"]
    assert "route-1" in route_failed[0].evidence["failed_routes"]

    ineligible = real._certificate_checks(_certification(routes=[_route(eligible="stress_only")]))
    assert [check.status for check in ineligible] == ["fail", "fail"]


def test_certificate_and_risk_helpers_handle_malformed_and_empty_evidence() -> None:
    """Malformed certificate payloads do not become accidental positive evidence."""
    malformed = CertificationStatus("scenario_cert.v1", "passed", "empty", {"certificates": []})
    checks = real._certificate_checks(malformed)
    assert [check.status for check in checks] == ["fail", "fail"]
    assert real._route_certificates(SimpleNamespace(details={"certificates": ["bad"]})) == []
    assert real._route_certificates(SimpleNamespace(details=None)) == []

    checks_with_unavailable = (
        FeasibilityCheck("kinematic_reachability", "unavailable", "missing", {}),
        FeasibilityCheck("behavioral_consistency", "fail", "bad", {"error": "bad"}),
        FeasibilityCheck("geometry_traffic", "unavailable", "missing", {}),
        FeasibilityCheck("simulator_validity", "unavailable", "missing", {}),
    )
    value, vector = real._risk_value(checks_with_unavailable, malformed, candidate_id="a" * 16)
    assert value.kinematic_criticality == 0.0
    assert len(vector) == 7


def test_behavior_check_and_simulator_check_fail_closed(tmp_path: Path) -> None:
    """Loader and benchmark evidence failures remain explicit and typed."""
    config = real._config_search(
        _MANIFEST,
        yaml.safe_load(_MANIFEST.read_text(encoding="utf-8")),
        output_dir=tmp_path / "out",
    )
    missing = real._behavior_check(tmp_path / "missing.yaml", config=config)
    assert missing.status == "unavailable"

    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("scenarios: []\n", encoding="utf-8")
    malformed = real._behavior_check(malformed_path, config=config)
    assert malformed.status == "fail"

    evaluation = SimpleNamespace(
        scenario_yaml_path=None,
        bundle_path=None,
        episode_record_path=None,
        objective_value=None,
        error="bad",
        failure_attribution=FailureAttribution("failed", "evaluation_error", ["bad"], {}),
    )
    not_run, runtime = real._simulator_check(
        evaluation, certification_status=_certification(status="failed")
    )
    assert not_run.status == "unavailable"
    assert runtime["execution_mode"] == "unknown"

    record_path = tmp_path / "episode.jsonl"
    record_path.write_text(json.dumps({"status": "success", "termination_reason": "goal"}) + "\n")
    evaluation.episode_record_path = record_path
    evaluation.failure_attribution = FailureAttribution(
        "attributed",
        "success",
        [],
        {"execution_mode": "native", "availability_status": "available"},
    )
    passed, runtime = real._simulator_check(evaluation, certification_status=_certification())
    assert passed.status == "pass"
    assert runtime["record_status"] == "success"

    evaluation.failure_attribution = FailureAttribution(
        "attributed",
        "unknown",
        [],
        {"execution_mode": "fallback", "availability_status": "available"},
    )
    unavailable, _runtime = real._simulator_check(evaluation, certification_status=_certification())
    assert unavailable.status == "unavailable"

    evaluation.failure_attribution = FailureAttribution(
        "attributed",
        "unknown",
        [],
        {"execution_mode": "native", "availability_status": "available"},
    )
    evaluation.episode_record_path = tmp_path / "missing_episode.jsonl"
    failed, _runtime = real._simulator_check(evaluation, certification_status=_certification())
    assert failed.status == "fail"

    evaluation.episode_record_path.write_text("not-json\n", encoding="utf-8")
    malformed_record, malformed_runtime = real._simulator_check(
        evaluation, certification_status=_certification()
    )
    assert malformed_record.status == "fail"
    assert "record_error" in malformed_runtime

    malformed_scenario = tmp_path / "malformed_metadata.yaml"
    malformed_scenario.write_text("scenarios:\n  - metadata: []\n", encoding="utf-8")
    malformed_behavior = real._behavior_check(malformed_scenario, config=config)
    assert malformed_behavior.status == "fail"


def test_method_summary_and_candidate_record_preserve_denominators() -> None:
    """Method summaries expose selected and safety denominators without rate claims."""
    first = _feasible_candidate("candidate-1")
    second = _feasible_candidate("candidate-2", family="other")
    runtime = {
        "candidate-1": {"execution_mode": "native", "primary_failure": "collision"},
        "candidate-2": {"execution_mode": "adapter", "primary_failure": "timeout"},
    }
    summary = real._method_summary([first, second], budget=2, threshold=0.2, runtime_by_id=runtime)
    assert summary["selected_count"] == 2
    assert summary["valid_scenario_rate"] == 1.0
    assert summary["safety_event_severity"]["denominator"] == 2
    assert summary["safety_event_severity"]["counts"] == {"collision": 1, "timeout": 1}
    assert real._unavailable_method("blocked")["status"] == "unavailable"
    assert real._candidate_record(first, runtime={"execution_mode": "native"})["runtime"] == {
        "execution_mode": "native"
    }
