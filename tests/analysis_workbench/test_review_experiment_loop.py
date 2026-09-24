"""Focused SREV-24 loop contract tests."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench import review_execute, review_experiment_loop
from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ComponentResult,
    SourceRef,
    experiment_recipe_canonical_digest,
)
from robot_sf.analysis_workbench.review_experiment_loop import (
    COMPONENT_ID,
    LEGACY_SESSION_JOURNAL_FILENAME,
    SESSION_JOURNAL_FILENAME,
    ExperimentLoop,
    ExperimentLoopError,
    LoopBudget,
    LoopPolicy,
    _answerability_document,
    _atomic_write_json,
    _bounded_int,
    _budget_from_config,
    _candidate_order,
    _canonical_digest,
    _control_fidelity,
    _finite_float,
    _has_valid_result_metrics,
    _is_negative_outcome,
    _json_bytes,
    _native_fidelity_attempts,
    _NativeExecutorAdapter,
    _operation_id,
    _pair_telemetry,
    _request_identity,
    _safe_text,
    _status_from_result,
    _validate_injected_source_proof,
    _validate_input,
    _validate_loop_budget,
    descriptor,
    main,
    run,
)


def _recipe(
    candidates: list[dict[str, Any]] | None = None,
    *,
    max_candidates: int = 3,
    max_executions: int = 6,
) -> dict[str, Any]:
    return {
        "schema_version": "experiment-recipe.v1",
        "recipe_id": "loop-test-recipe",
        "hypothesis": "A diagnostic intervention changes the recorded metric.",
        "source_identity": {"scenario_id": "loop-fixture"},
        "interventions": candidates
        or [
            {
                "intervention_id": "low",
                "factor": "single_pedestrian_speed_offset",
                "priority": 2,
            },
            {
                "intervention_id": "high",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            },
            {
                "intervention_id": "tie-id",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            },
        ],
        "control_conditions": {"ped_speed_m_s": 1.0},
        "measurements": [
            {
                "name": "min_robot_ped_distance_m",
                "units": "metres",
                "expected_direction": "increase",
            }
        ],
        "budget": {
            "max_candidates": max_candidates,
            "max_executions": max_executions,
            "wall_timeout_s": 600,
        },
        "stop_rules": [
            "exhausted_candidates",
            "execution_budget_exhausted",
            "wall_timeout",
            "control_fidelity_failure_blocks_treatment",
        ],
        "preservation_destination": "external:loop-test",
    }


def _request(recipe: dict[str, Any], *, output: str = "loop") -> ComponentRequest:
    return ComponentRequest(
        request_id="loop-request",
        component_id=COMPONENT_ID,
        sources=(SourceRef("loop-source", "source.json", "fixture.v1"),),
        output_directory=output,
        config={"recipe": recipe},
    )


def _metrics(kind: str, *, value: float = 1.0) -> dict[str, Any]:
    treatment = kind == "treatment"
    return {
        "ped_displacement_m": 1.0,
        "robot_displacement_m": 1.0,
        "ped_mean_speed_m_s": 1.0 if not treatment else 1.2,
        "min_robot_ped_distance_m": value if treatment else 1.0,
        "robot_goal_reached": 1,
        "ped_motion_onset_step": 1,
    }


def _source_proof(base: Path, request: ComponentRequest) -> dict[str, Any]:
    """Build a measured, base-bound proof for the injected test executor."""

    recipe = request.config["recipe"]
    source_root = base / "admission-root"
    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / "source.json"
    source_path.write_text('{"fixture": "loop"}\n', encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = request.sources[0]
    return {
        "status": "admitted",
        "receipt_id": "fixture",
        "source_root": str(source_root),
        "source": {
            "artifact_id": source.artifact_id,
            "uri": source.uri,
            "format": source.format,
            "sha256": digest,
        },
        "request_digest": _canonical_digest(_request_identity(request)),
        "recipe_digest": experiment_recipe_canonical_digest(recipe),
        "evidence_boundary": "diagnostic_only",
        "scientific_claim_allowed": False,
        "dependent_family_status": "standalone_fixture_only",
    }


class FakeExecutor:
    """Small idempotent executor with explicit call accounting."""

    def __init__(self, *, values: dict[str, float] | None = None) -> None:
        """Initialize empty call and idempotency ledgers."""
        self.calls: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.values = values or {}

    def execute(
        self,
        operation_id: str,
        candidate: dict[str, Any],
        kind: str,
        spec: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        del spec
        self.calls.append(
            {
                "operation_id": operation_id,
                "candidate": candidate["intervention_id"],
                "kind": kind,
                "attempt": attempt,
            }
        )
        result = {
            "status": "ok",
            "metrics": _metrics(kind, value=self.values.get(candidate["intervention_id"], 1.2)),
            "mechanism_activated": True,
        }
        self.results[operation_id] = result
        return result

    def result_for(self, operation_id: str) -> dict[str, Any] | None:
        return self.results.get(operation_id)


class _FakeMonotonicClock:
    """Deterministic monotonic clock for phase-boundary deadline tests."""

    def __init__(self) -> None:
        """Start the clock at the beginning of the test."""
        self.value = 0.0

    def __call__(self) -> float:
        """Return the current synthetic monotonic time."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Advance time only at the phase boundary under test."""
        self.value += seconds


def _run_injected(
    request: ComponentRequest,
    *,
    base: Path,
    executor: Any,
    resume: bool = False,
    autonomous: bool = True,
    cancel: Any = None,
):
    return run(
        request,
        base=base,
        autonomous=autonomous,
        resume=resume,
        executor=executor,
        source_admission=_source_proof(base, request),
        cancel=cancel,
    )


def _native_fixture_request(
    output: str,
    *,
    max_candidates: int = 1,
    max_executions: int = 2,
) -> tuple[ComponentRequest, dict[str, Any]]:
    fixture_root = Path("tests/fixtures/scenario_review/review_experiment_loop")
    request_payload = json.loads((fixture_root / "request.json").read_text())
    config = json.loads((fixture_root / "config.json").read_text())
    admission = json.loads((fixture_root / "admission.json").read_text())
    request_payload["config"] = {
        **config,
        "max_candidates": max_candidates,
        "max_executions": max_executions,
        "executor_config": {**config["executor_config"], "horizon_steps": 1},
    }
    request_payload["output_directory"] = output
    request = ComponentRequest(
        request_payload["request_id"],
        request_payload["component_id"],
        (SourceRef(**request_payload["sources"][0]),),
        request_payload["output_directory"],
        request_payload["config"],
    )
    return request, admission


def _write_valid_native_pair(
    child_request: ComponentRequest,
    base: Path,
    admission_config: Any,
) -> None:
    """Write the supported SREV-22 child envelope used by recovery tests."""

    effective_config = {
        **child_request.config,
        "admission": admission_config.to_dict(),
    }
    validated = review_execute.validate_execute_config(effective_config)
    proof, failure = review_execute._resolve_executor_admission(
        child_request,
        validated,
        validated.recipe,
        admission=admission_config,
    )
    assert proof is not None, failure
    source_admission = proof.to_dict()
    os.close(proof.root_fd)
    control_metrics = _metrics("control")
    treatment_metrics = _metrics("treatment", value=1.2)
    report = {
        "intervention_id": "ped-speed-up",
        "factor": "single_pedestrian_speed_offset",
        "status": "complete",
        "verdict": "survived",
        "verdict_reason": "mechanism activated and min_robot_ped_distance_m moved increase by +0.2",
        "control_metrics": control_metrics,
        "treatment_metrics": treatment_metrics,
        "control_activated": True,
        "treatment_activated": True,
        "nonintervened_config_match": True,
    }
    attempts = [
        {
            "candidate_id": "ped-speed-up",
            "kind": "control",
            "status": "ok",
            "elapsed_s": 0.001,
            "metrics": control_metrics,
        },
        {
            "candidate_id": "ped-speed-up",
            "kind": "treatment",
            "status": "ok",
            "elapsed_s": 0.001,
            "metrics": treatment_metrics,
        },
    ]
    traces = [
        {
            "schema_version": review_execute.ACTIVATION_TRACE_SCHEMA_VERSION,
            "intervention_id": "ped-speed-up",
            "factor": "single_pedestrian_speed_offset",
            "control_activated": True,
            "treatment_activated": True,
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
        }
    ]
    ledger = {
        "schema_version": review_execute.ATTEMPT_LEDGER_SCHEMA_VERSION,
        "request_id": child_request.request_id,
        "component_id": review_execute.COMPONENT_ID,
        "recipe_id": validated.recipe["recipe_id"],
        "request_digest": review_execute._canonical_digest(
            review_execute._request_identity_document(child_request)
        ),
        "recipe_digest": review_execute._canonical_digest(validated.recipe),
        "config_identity_digest": review_execute._canonical_digest(
            review_execute._config_identity_document(validated)
        ),
        "budget": review_execute._config_budget_document(validated),
        "attempts": attempts,
        "executions_consumed": 2,
        "candidate_reports": [report],
        "traces": traces,
        "evidence_boundary": "diagnostic_only",
        "scientific_claim_allowed": False,
        "dependent_family_status": "standalone_fixture_only",
        "source_admission": source_admission,
        "wall_elapsed_s": 0.001,
    }
    provenance = review_execute._commit_provenance()
    provenance.update(
        {
            "recipe_id": validated.recipe["recipe_id"],
            "source_identity": dict(validated.recipe["source_identity"]),
            "request_digest": ledger["request_digest"],
            "recipe_digest": ledger["recipe_digest"],
            "config_digest": review_execute._canonical_digest(
                review_execute._config_document(validated)
            ),
            "source_admission": source_admission,
            "sources": [
                {
                    "artifact_id": source.artifact_id,
                    "uri": source.uri,
                    "format": source.format,
                }
                for source in child_request.sources
            ],
            "output_directory": child_request.output_directory,
        }
    )
    execute_report = {
        "schema_version": review_execute.EXECUTE_REPORT_SCHEMA_VERSION,
        "request_id": child_request.request_id,
        "component_id": review_execute.COMPONENT_ID,
        "recipe_id": validated.recipe["recipe_id"],
        "source_identity": dict(validated.recipe["source_identity"]),
        "source_admission": source_admission,
        "evidence_boundary": "diagnostic_only",
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": "standalone_fixture_only",
        "candidates": [report],
        "budget": {
            **review_execute._config_budget_document(validated),
            "executions_consumed": 2,
            "wall_elapsed_s": 0.001,
        },
        "provenance": provenance,
    }
    child_dir = base / "executor"
    child_dir.mkdir(parents=True, exist_ok=True)
    (child_dir / "attempt-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    (child_dir / "execute-report.json").write_text(json.dumps(execute_report), encoding="utf-8")


def test_descriptor_and_priority_then_id_ordering(tmp_path: Path) -> None:
    assert descriptor()["component_id"] == COMPONENT_ID
    assert descriptor()["output_types"] == [
        "experiment-loop-report.v1",
        "experiment-loop-session.v1",
    ]
    recipe = _recipe(
        [
            {"intervention_id": "z", "factor": "single_pedestrian_speed_offset", "priority": 1},
            {"intervention_id": "a", "factor": "single_pedestrian_speed_offset", "priority": 1},
        ],
        max_candidates=2,
        max_executions=4,
    )
    fake = FakeExecutor()
    result = _run_injected(
        _request(recipe),
        base=tmp_path,
        executor=fake,
    )
    assert result.status == "complete"
    assert [call["candidate"] for call in fake.calls] == ["a", "a", "z", "z"]
    report = json.loads((tmp_path / "loop" / "experiment-loop-report.json").read_text())
    assert [item["intervention_id"] for item in report["candidate_order"]] == ["a", "z"]


def test_read_only_and_unauthorised_policy_never_call_executor(tmp_path: Path) -> None:
    fake = FakeExecutor()
    recipe = _recipe(max_candidates=1, max_executions=2)
    assert (
        run(_request(recipe), base=tmp_path / "read", read_only=True, executor=fake).reason
        == "read_only_never_executes"
    )
    assert (
        run(_request(recipe), base=tmp_path / "auth", executor=fake).reason
        == "autonomous_start_authorization_required"
    )
    assert fake.calls == []


def test_strict_serialization_and_scalar_guards_fail_closed() -> None:
    with pytest.raises(ExperimentLoopError, match="strict JSON"):
        _canonical_digest({"not_json": float("nan")})
    with pytest.raises(ExperimentLoopError, match="strict JSON"):
        _json_bytes({"not_json": float("inf")})
    with pytest.raises(ExperimentLoopError, match="non-empty"):
        _safe_text("   ", field_name="session_id")
    with pytest.raises(ExperimentLoopError, match="NUL"):
        _safe_text("bad\x00value", field_name="session_id")
    with pytest.raises(ExperimentLoopError, match="invalid Unicode"):
        _safe_text("\ud800", field_name="session_id")
    with pytest.raises(ExperimentLoopError, match="finite number"):
        _finite_float(True, field_name="wall_timeout_s")
    with pytest.raises(ExperimentLoopError, match="finite and >="):
        _finite_float(float("nan"), field_name="wall_timeout_s")
    with pytest.raises(ExperimentLoopError, match="finite and >="):
        _finite_float(-1.0, field_name="wall_timeout_s")
    with pytest.raises(ExperimentLoopError, match="must be an integer"):
        _bounded_int(False, field_name="max_candidates", minimum=1, maximum=3)
    with pytest.raises(ExperimentLoopError, match="within"):
        _bounded_int(4, field_name="max_candidates", minimum=1, maximum=3)


def test_atomic_json_persistence_rejects_symlink_and_cleans_failed_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(OSError, match="symlink"):
        _atomic_write_json(linked_root / "journal.json", {})

    target = real_root / "target.json"
    target.write_text("{}", encoding="utf-8")
    temporary = real_root / ".target.json.tmp"
    temporary.symlink_to(target)
    with pytest.raises(OSError, match="symlink"):
        _atomic_write_json(target, {})
    temporary.unlink()

    original_fdopen = review_experiment_loop.os.fdopen

    def fail_fdopen(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("synthetic fdopen failure")

    monkeypatch.setattr(review_experiment_loop.os, "fdopen", fail_fdopen)
    with pytest.raises(RuntimeError, match="fdopen"):
        _atomic_write_json(real_root / "failed.json", {})
    monkeypatch.setattr(review_experiment_loop.os, "fdopen", original_fdopen)

    original_open = review_experiment_loop.os.open

    def fail_directory_open(path: Any, flags: int, *args: Any) -> int:
        if Path(path) == real_root:
            raise OSError("synthetic directory fsync failure")
        return original_open(path, flags, *args)

    monkeypatch.setattr(review_experiment_loop.os, "open", fail_directory_open)
    _atomic_write_json(real_root / "no-directory-fsync.json", {})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda proof: proof.clear(),
        lambda proof: proof.update(status="rejected"),
        lambda proof: proof.update(evidence_boundary="benchmark"),
        lambda proof: proof.update(scientific_claim_allowed=True),
        lambda proof: proof.update(dependent_family_status="shared"),
        lambda proof: proof.update(request_digest="0" * 64),
        lambda proof: proof.update(recipe_digest="0" * 64),
        lambda proof: proof.update(source=[]),
        lambda proof: proof.update(source_root=""),
        lambda proof: proof["source"].update(uri=""),
        lambda proof: proof["source"].update(sha256="bad"),
        lambda proof: proof["source"].update(artifact_id=None),
        lambda proof: proof["source"].update(sha256="0" * 64),
    ],
    ids=[
        "non-mapping",
        "status",
        "boundary",
        "scientific-claim",
        "dependent-family",
        "request-digest",
        "recipe-digest",
        "source-shape",
        "source-root",
        "source-uri",
        "source-digest-shape",
        "source-identity-shape",
        "source-digest-mismatch",
    ],
)
def test_injected_admission_proof_rejects_untrusted_shapes(tmp_path: Path, mutation: Any) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2))
    proof = _source_proof(tmp_path, request)
    mutation(proof)
    assert (
        _validate_injected_source_proof(
            request,
            request.config["recipe"],
            proof,
            base=tmp_path,
        )
        is None
    )


def test_injected_admission_proof_rejects_path_and_binding_mismatches(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2))
    proof = _source_proof(tmp_path, request)
    proof["source"]["uri"] = "/etc/passwd"
    assert (
        _validate_injected_source_proof(request, request.config["recipe"], proof, base=tmp_path)
        is None
    )

    proof = _source_proof(tmp_path, request)
    proof["source"]["artifact_id"] = "unknown"
    assert (
        _validate_injected_source_proof(request, request.config["recipe"], proof, base=tmp_path)
        is None
    )

    proof = _source_proof(tmp_path, request)
    source_path = tmp_path / "admission-root" / "source.json"
    source_path.unlink()
    source_path.symlink_to(tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text("outside", encoding="utf-8")
    assert (
        _validate_injected_source_proof(request, request.config["recipe"], proof, base=tmp_path)
        is None
    )

    recipe = _recipe(max_candidates=1, max_executions=2)
    recipe["source_identity"]["source_ref"] = {
        "artifact_id": "different",
        "uri": "source.json",
        "format": "fixture.v1",
    }
    bound_request = _request(recipe)
    bound_proof = _source_proof(tmp_path / "bound", bound_request)
    assert (
        _validate_injected_source_proof(bound_request, recipe, bound_proof, base=tmp_path) is None
    )


def test_candidate_and_budget_validation_rejects_unsafe_limits() -> None:
    with pytest.raises(ExperimentLoopError, match="non-empty list"):
        _candidate_order({"interventions": []})
    with pytest.raises(ExperimentLoopError, match="must be a mapping"):
        _candidate_order({"interventions": [None]})
    with pytest.raises(ExperimentLoopError, match="duplicate"):
        _candidate_order(
            {
                "interventions": [
                    {"intervention_id": "same"},
                    {"intervention_id": "same"},
                ]
            }
        )
    with pytest.raises(ExperimentLoopError, match="priority"):
        _candidate_order({"interventions": [{"intervention_id": "bad", "priority": -1}]})

    with pytest.raises(ExperimentLoopError, match="budget must be a mapping"):
        _budget_from_config({}, {"budget": None})
    recipe = _recipe(max_candidates=1, max_executions=2)
    recipe["budget"]["wall_timeout_s"] = 601
    with pytest.raises(ExperimentLoopError, match="exceeds 600"):
        _budget_from_config({}, recipe)
    recipe = _recipe(max_candidates=1, max_executions=2)
    with pytest.raises(ExperimentLoopError, match="max_candidates exceeds"):
        _budget_from_config({"max_candidates": 2}, recipe)
    with pytest.raises(ExperimentLoopError, match="max_executions exceeds"):
        _budget_from_config({"max_executions": 3}, recipe)
    with pytest.raises(ExperimentLoopError, match="wall_timeout_s exceeds"):
        _budget_from_config({"wall_timeout_s": 601}, _recipe(max_candidates=1, max_executions=2))


def test_validate_input_rejects_malformed_policy_and_answerability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validate(config: Any) -> None:
        request = ComponentRequest(
            "invalid-request",
            COMPONENT_ID,
            (SourceRef("loop-source", "source.json", "fixture.v1"),),
            "invalid",
            config,
        )
        _validate_input(request, autonomous=True, read_only=False, resume=False)

    with pytest.raises(ExperimentLoopError, match="config must be a mapping"):
        validate([])
    with pytest.raises(ExperimentLoopError, match="unknown keys"):
        validate({"recipe": _recipe(max_candidates=1, max_executions=2), "unexpected": True})
    with pytest.raises(ExperimentLoopError, match="config.recipe"):
        validate({"recipe": None})
    with pytest.raises(ExperimentLoopError, match="corrupt_recipe"):
        validate({"recipe": {"schema_version": "experiment-recipe.v1"}})
    base = {"recipe": _recipe(max_candidates=1, max_executions=2)}
    for key, value, message in (
        ("mode", "manual", "mode"),
        ("autonomous", 1, "autonomous"),
        ("read_only", 1, "read_only"),
        ("cancel_requested", 1, "cancel_requested"),
        ("session_id", "", "session_id"),
        ("executor_config", [], "executor_config"),
        ("answerability", [], "answerability"),
    ):
        config = {**base, key: value}
        with pytest.raises(ExperimentLoopError, match=message):
            validate(config)

    def fail_answerability(_value: Any) -> Any:
        raise ValueError("synthetic answerability failure")

    monkeypatch.setattr(review_experiment_loop, "evaluate_answerability", fail_answerability)
    with pytest.raises(ExperimentLoopError, match="invalid_answerability"):
        validate({**base, "answerability": {}})


def test_answerability_document_uses_evaluator_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class Evaluated:
        def as_dict(self) -> dict[str, str]:
            return {"state": "diagnostic_only"}

    monkeypatch.setattr(
        review_experiment_loop, "evaluate_answerability", lambda _value: Evaluated()
    )
    assert _answerability_document({"answerability": "fixture"}) == {"state": "diagnostic_only"}


def test_status_normalization_preserves_supported_executor_shapes() -> None:
    component = ComponentResult("request", COMPONENT_ID, "complete")
    assert _status_from_result(component)[0] == "ok"
    assert _status_from_result(object())[0] == "failed"
    assert _status_from_result({"result": {"status": "ok", "metrics": {"x": 1}}})[0] == "ok"
    assert _status_from_result({"status": "terminal"})[1]["terminal"] is True
    assert _status_from_result({"status": "canceled"})[0] == "cancelled"
    assert _status_from_result({"status": "not_available"})[0] == "unavailable"
    assert _status_from_result({"status": "timed_out"})[0] == "failed"
    assert _status_from_result({"metrics": {"x": 1}})[0] == "ok"
    assert _status_from_result({"detail": "missing status"})[0] == "failed"


def test_pair_telemetry_and_fidelity_fail_closed_on_missing_or_invalid_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = {
        "metrics": {"ped_displacement_m": 1.0, "robot_displacement_m": 1.0},
        "activation": {"activated": True},
    }
    treatment = {
        "metrics": {"ped_displacement_m": 1.0, "robot_displacement_m": 1.0},
        "activation": {"activated": True},
    }
    measurement = {"name": "min_robot_ped_distance_m", "expected_direction": "increase"}
    missing = _pair_telemetry(
        control,
        treatment,
        factor="single_pedestrian_speed_offset",
        measurement=measurement,
        motion_epsilon=0.05,
    )
    assert missing["outcome"] == "inconclusive"
    assert missing["reason"].startswith("measurement_missing:")

    def fail_pair(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("synthetic pair failure")

    monkeypatch.setattr(review_experiment_loop, "evaluate_counterfactual_pair", fail_pair)
    control["metrics"]["min_robot_ped_distance_m"] = 1.0
    treatment["metrics"]["min_robot_ped_distance_m"] = 1.2
    unavailable = _pair_telemetry(
        control,
        treatment,
        factor="single_pedestrian_speed_offset",
        measurement=measurement,
        motion_epsilon=0.05,
    )
    assert unavailable["outcome"] == "inconclusive"
    assert unavailable["reason"].startswith("pair_evaluation_unavailable:")

    class UnknownVerdict:
        verdict = "unknown"
        reason = "synthetic verdict"

    monkeypatch.setattr(
        review_experiment_loop, "evaluate_counterfactual_pair", lambda *a, **k: UnknownVerdict()
    )
    inconclusive = _pair_telemetry(
        control,
        treatment,
        factor="single_pedestrian_speed_offset",
        measurement=measurement,
        motion_epsilon=0.05,
    )
    assert inconclusive["outcome"] == "inconclusive"

    assert not _control_fidelity(
        {"fidelity": {"status": "failed", "reason": "child check"}}, motion_epsilon=0.05
    )[0]
    assert not _control_fidelity({"control_fidelity": "invalid"}, motion_epsilon=0.05)[0]
    assert not _control_fidelity({"metrics": {"ped_displacement_m": 0.0}}, motion_epsilon=0.05)[0]
    assert _native_fidelity_attempts({"native_fidelity_attempts": True}) is None
    assert not _has_valid_result_metrics({"status": "failed"})


def test_direct_loop_budget_rejects_ceiling_and_process_mismatches() -> None:
    budget = LoopBudget(max_candidates=1, max_executions=2, wall_timeout_s=600)
    with pytest.raises(ExperimentLoopError, match="budget must be a mapping"):
        _validate_loop_budget({"budget": None}, budget)
    with pytest.raises(ExperimentLoopError, match="wall timeout exceeds"):
        _validate_loop_budget(
            {"budget": {"max_candidates": 1, "max_executions": 2, "wall_timeout_s": 600}},
            LoopBudget(1, 2, 601),
        )
    with pytest.raises(ExperimentLoopError, match="only one local CPU"):
        _validate_loop_budget(
            {"budget": {"max_candidates": 1, "max_executions": 2, "wall_timeout_s": 600}},
            LoopBudget(1, 2, 600, max_concurrent_local_cpu_processes=2),
        )
    with pytest.raises(ExperimentLoopError, match="exceeds recipe budget"):
        _validate_loop_budget(
            {"budget": {"max_candidates": 1, "max_executions": 2, "wall_timeout_s": 600}},
            LoopBudget(2, 2, 600),
        )


def test_cli_read_only_and_malformed_input_are_truthful(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_root = Path("tests/fixtures/scenario_review/review_experiment_loop")
    result = main(
        [
            "--input",
            str(fixture_root / "request.json"),
            "--config",
            str(fixture_root / "config.json"),
            "--output",
            "cli-read-only",
            "--base",
            str(tmp_path),
            "--read-only",
        ]
    )
    assert result == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert main(["--input", str(invalid), "--output", "invalid"]) == 1
    assert "cannot be parsed safely" in capsys.readouterr().out


def test_pair_reservation_prevents_partial_new_pair(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=2, max_executions=3)
    fake = FakeExecutor()
    result = _run_injected(
        _request(recipe),
        base=tmp_path,
        executor=fake,
    )
    assert result.status == "partial"
    assert result.reason == "execution_budget_exhausted"
    assert len(fake.calls) == 2
    journal = json.loads((tmp_path / "loop" / SESSION_JOURNAL_FILENAME).read_text())
    assert journal["executions_consumed"] == 2
    assert journal["reserved_executions"] == 0
    assert journal["accounting"]["controls"] == 1
    assert journal["accounting"]["treatments"] == 1


def test_resume_extends_pair_budget_without_resetting_consumed_attempts(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=2, max_executions=4)
    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "loop",
        {"recipe": recipe, "max_candidates": 2, "max_executions": 2},
    )
    first_executor = FakeExecutor()
    first = _run_injected(
        first_request,
        base=tmp_path,
        executor=first_executor,
    )
    assert first.status == "partial"
    assert first.reason == "execution_budget_exhausted"

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "loop",
        {"recipe": recipe, "max_candidates": 2, "max_executions": 4},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete"
    assert [call["candidate"] for call in resumed_executor.calls] == ["tie-id", "tie-id"]
    journal = json.loads((tmp_path / "loop" / SESSION_JOURNAL_FILENAME).read_text())
    assert journal["executions_consumed"] == 4


def test_execution_budget_limited_retry_resumes_without_terminal_failure(
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        [
            {
                "intervention_id": "budget-retry",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            }
        ],
        max_candidates=1,
        max_executions=4,
    )

    class RetryThenSuccess(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            if kind == "control" and attempt == 1:
                result["status"] = "failed"
                result["retryable"] = True
                result["reason"] = "transient"
            return result

    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "budget-retry",
        {"recipe": recipe, "max_executions": 2, "max_retries": 1},
    )
    first_executor = RetryThenSuccess()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "partial"
    assert first.reason == "execution_budget_exhausted"
    assert [call["attempt"] for call in first_executor.calls] == [1, 2]
    journal_path = tmp_path / "budget-retry" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["candidates"]["budget-retry"]["state"] == "incomplete"
    assert journal["candidates"]["budget-retry"]["incomplete_reason"] == (
        "execution_budget_exhausted"
    )
    assert journal["outcomes"] == []
    assert journal["stop_reason"] == "execution_budget_exhausted"
    assert journal["executions_consumed"] == 2
    assert journal["accounting"] == {
        "controls": 2,
        "treatments": 0,
        "failures": 1,
        "retries": 1,
        "fidelity_attempts": 1,
    }

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "budget-retry",
        {"recipe": recipe, "max_executions": 4, "max_retries": 1},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete", resumed.reason
    assert [(call["kind"], call["attempt"]) for call in resumed_executor.calls] == [
        ("treatment", 1)
    ]
    resumed_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert resumed_journal["executions_consumed"] == 3
    assert len(resumed_journal["operations"]) == 3
    assert resumed_journal["candidates"]["budget-retry"]["state"] == "complete"
    assert len(resumed_journal["outcomes"]) == 1


def test_resume_partial_pair_reserves_only_remaining_retry_slot(tmp_path: Path) -> None:
    recipe = _recipe(
        [
            {
                "intervention_id": "partial-pair",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            }
        ],
        max_candidates=1,
        max_executions=4,
    )

    class RetryEachSideOnce(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            if attempt == 1:
                result.update(status="failed", retryable=True, reason=f"{kind} transient")
            return result

    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "partial-pair",
        {"recipe": recipe, "max_executions": 3, "max_retries": 1},
    )
    first_executor = RetryEachSideOnce()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "partial"
    assert first.reason == "execution_budget_exhausted"
    assert [(call["kind"], call["attempt"]) for call in first_executor.calls] == [
        ("control", 1),
        ("control", 2),
        ("treatment", 1),
    ]
    journal_path = tmp_path / "partial-pair" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["candidates"]["partial-pair"]["state"] == "incomplete"
    assert journal["executions_consumed"] == 3
    assert journal["reserved_executions"] == 0

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "partial-pair",
        {"recipe": recipe, "max_executions": 4, "max_retries": 1},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete", resumed.reason
    assert [(call["kind"], call["attempt"]) for call in resumed_executor.calls] == [
        ("treatment", 2)
    ]
    resumed_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert resumed_journal["executions_consumed"] == 4
    assert resumed_journal["accounting"] == {
        "controls": 2,
        "treatments": 2,
        "failures": 2,
        "retries": 2,
        "fidelity_attempts": 1,
    }


def test_resume_widens_exhausted_candidate_ceiling_without_restarting_prefix(
    tmp_path: Path,
) -> None:
    recipe = _recipe(max_candidates=2, max_executions=4)
    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "widen-candidates",
        {"recipe": recipe, "max_candidates": 1, "max_executions": 2},
    )
    first_executor = FakeExecutor()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "complete"
    assert first.reason == "exhausted_candidates"
    assert [call["candidate"] for call in first_executor.calls] == ["high", "high"]

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "widen-candidates",
        {"recipe": recipe, "max_candidates": 2, "max_executions": 4},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete"
    assert [call["candidate"] for call in resumed_executor.calls] == ["tie-id", "tie-id"]
    journal = json.loads(
        (tmp_path / "widen-candidates" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert [item["intervention_id"] for item in journal["outcomes"]] == ["high", "tie-id"]
    assert journal["executions_consumed"] == 4


def test_resume_widens_retry_ceiling_for_retryable_failed_outcome(tmp_path: Path) -> None:
    recipe = _recipe(
        [{"intervention_id": "retry", "factor": "single_pedestrian_speed_offset", "priority": 1}],
        max_candidates=1,
        max_executions=4,
    )
    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "widen-retries",
        {"recipe": recipe, "max_retries": 0},
    )

    class RetryableFailure(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            del candidate, spec
            self.calls.append({"operation_id": operation_id, "kind": kind, "attempt": attempt})
            return {"status": "failed", "retryable": True, "reason": "transient"}

    first = _run_injected(
        first_request,
        base=tmp_path,
        executor=RetryableFailure(),
    )
    assert first.status == "failed"
    assert first.reason == "candidate_execution_failed"

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "widen-retries",
        {"recipe": recipe, "max_retries": 1},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete"
    assert [call["kind"] for call in resumed_executor.calls] == ["control", "treatment"]
    journal = json.loads(
        (tmp_path / "widen-retries" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert journal["accounting"]["retries"] == 1
    assert journal["accounting"]["failures"] == 1
    assert journal["executions_consumed"] == 3


def test_resume_does_not_reopen_when_final_retry_attempt_is_permanent(
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        [
            {
                "intervention_id": "retry-final",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            }
        ],
        max_candidates=1,
        max_executions=4,
    )
    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "retry-final",
        {"recipe": recipe, "max_retries": 1},
    )

    class RetryThenPermanent(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            del candidate, spec
            self.calls.append({"operation_id": operation_id, "kind": kind, "attempt": attempt})
            if kind == "control" and attempt == 1:
                return {"status": "failed", "retryable": True, "reason": "transient"}
            return {"status": "failed", "reason": "permanent"}

    first_executor = RetryThenPermanent()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "failed"
    assert first.reason == "candidate_execution_failed"
    assert [call["attempt"] for call in first_executor.calls] == [1, 2]

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "retry-final",
        {"recipe": recipe, "max_retries": 2},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "failed"
    assert resumed.reason == "candidate_execution_failed"
    assert resumed_executor.calls == []
    journal = json.loads(
        (tmp_path / "retry-final" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert journal["candidates"]["retry-final"]["state"] == "failed"
    assert len(journal["outcomes"]) == 1


def test_retry_and_fidelity_attempts_are_accounted(tmp_path: Path) -> None:
    recipe = _recipe(
        [{"intervention_id": "retry", "factor": "single_pedestrian_speed_offset", "priority": 1}],
        max_candidates=1,
        max_executions=4,
    )

    class RetryExecutor(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            if len(self.calls) == 0:
                self.calls.append({"operation_id": operation_id, "kind": kind, "attempt": attempt})
                return {"status": "failed", "retryable": True, "reason": "transient"}
            return super().execute(operation_id, candidate, kind, spec, attempt)

    request = _request(recipe)
    request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "max_retries": 1},
    )
    executor = RetryExecutor()
    result = _run_injected(
        request,
        base=tmp_path,
        executor=executor,
    )
    assert result.status == "complete"
    journal = json.loads((tmp_path / "loop" / SESSION_JOURNAL_FILENAME).read_text())
    assert journal["accounting"] == {
        "controls": 2,
        "treatments": 1,
        "failures": 1,
        "retries": 1,
        "fidelity_attempts": 1,
    }
    journal["outcomes"][0]["operation_ids"] = [
        next(
            operation["operation_id"]
            for operation in reversed(journal["operations"])
            if operation["kind"] == kind
        )
        for kind in ("control", "treatment")
    ]
    (tmp_path / "loop" / SESSION_JOURNAL_FILENAME).write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "complete"


def _delete_retry_predecessor(journal: dict[str, Any]) -> None:
    operations = journal["operations"]
    predecessor = next(
        operation
        for operation in operations
        if operation["kind"] == "control" and operation["attempt"] == 1
    )
    operations.remove(predecessor)
    candidate = journal["candidates"]["retry"]
    candidate["operation_ids"].remove(predecessor["operation_id"])
    candidate["attempts"] -= 1
    journal["outcomes"][0]["operation_ids"].remove(predecessor["operation_id"])
    journal["executions_consumed"] -= 1
    journal["accounting"]["controls"] -= 1
    journal["accounting"]["failures"] -= 1


def _retry_after_success(journal: dict[str, Any]) -> None:
    predecessor = next(
        operation
        for operation in journal["operations"]
        if operation["kind"] == "control" and operation["attempt"] == 1
    )
    predecessor["state"] = "completed"
    predecessor["result"] = {"status": "ok", "metrics": _metrics("control")}
    predecessor["failure_accounted"] = False
    journal["accounting"]["failures"] = 0
    journal["accounting"]["fidelity_attempts"] = 2


@pytest.mark.parametrize(
    "mutation",
    [_delete_retry_predecessor, _retry_after_success],
    ids=["deleted-predecessor", "retry-after-success"],
)
def test_resume_rejects_noncontiguous_or_invalid_retry_history(
    tmp_path: Path, mutation: Any
) -> None:
    recipe = _recipe(
        [{"intervention_id": "retry", "factor": "single_pedestrian_speed_offset", "priority": 1}],
        max_candidates=1,
        max_executions=4,
    )
    request = _request(recipe, output="retry-history")
    request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "max_retries": 1},
    )

    class RetryForValidation(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            if not self.calls:
                self.calls.append({"operation_id": operation_id, "kind": kind, "attempt": attempt})
                return {"status": "failed", "retryable": True, "reason": "transient"}
            return super().execute(operation_id, candidate, kind, spec, attempt)

    _run_injected(request, base=tmp_path, executor=RetryForValidation())
    journal_path = tmp_path / "retry-history" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    mutation(journal)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "retry" in resumed.reason


def test_metricless_success_is_failed_and_immediately_resumable(tmp_path: Path) -> None:
    class Metricless(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            del candidate, kind, spec, attempt
            self.calls.append({"operation_id": operation_id})
            return {"status": "ok"}

    request = _request(_recipe(max_candidates=1, max_executions=2), output="metricless")
    first = _run_injected(request, base=tmp_path, executor=Metricless())
    assert first.status == "failed"
    journal = json.loads(
        (tmp_path / "metricless" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert journal["status"] == "failed"
    assert journal["operations"][0]["state"] == "failed"
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "failed"
    assert "cannot resume" not in resumed.reason
    assert resumed_executor.calls == []


def test_cancellation_after_control_does_not_dispatch_treatment(tmp_path: Path) -> None:
    executor = FakeExecutor()
    result = _run_injected(
        _request(_recipe(max_candidates=1, max_executions=2)),
        base=tmp_path,
        executor=executor,
        cancel=lambda: bool(executor.calls),
    )
    assert result.status == "cancelled"
    assert [call["kind"] for call in executor.calls] == ["control"]


def test_negative_outcomes_are_retained(tmp_path: Path) -> None:
    recipe = _recipe(
        [
            {
                "intervention_id": "falsified",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            },
            {
                "intervention_id": "inconclusive",
                "factor": "single_pedestrian_speed_offset",
                "priority": 2,
            },
        ],
        max_candidates=2,
        max_executions=4,
    )

    class Outcomes(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            if candidate["intervention_id"] == "inconclusive" and kind == "treatment":
                result["mechanism_activated"] = False
            elif candidate["intervention_id"] == "falsified" and kind == "treatment":
                result["metrics"]["min_robot_ped_distance_m"] = 0.5
            return result

    result = _run_injected(
        _request(recipe),
        base=tmp_path,
        executor=Outcomes(),
    )
    assert result.status == "complete"
    report = json.loads((tmp_path / "loop" / "experiment-loop-report.json").read_text())
    assert [item["outcome"] for item in report["outcomes"]] == ["falsified", "inconclusive"]
    assert len(report["negative_outcomes"]) == 2


def test_negative_export_is_derived_from_status_not_mutable_flag() -> None:
    assert _is_negative_outcome({"status": "failed", "negative": False}) is True
    assert _is_negative_outcome({"status": "unavailable", "negative": False}) is True
    assert _is_negative_outcome({"status": "cancelled", "negative": False}) is True
    assert (
        _is_negative_outcome({"status": "complete", "outcome": "survived", "negative": True})
        is False
    )


def test_sparse_metrics_without_explicit_activation_are_unavailable(
    tmp_path: Path,
) -> None:
    class SparseActivation(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            result.pop("mechanism_activated")
            return result

    request = _request(_recipe(max_candidates=1, max_executions=2), output="activation-missing")
    result = _run_injected(request, base=tmp_path, executor=SparseActivation())
    assert result.status == "unavailable"
    report = json.loads(
        (tmp_path / "activation-missing" / "experiment-loop-report.json").read_text(
            encoding="utf-8"
        )
    )
    outcome = report["outcomes"][0]
    assert outcome["status"] == "unavailable"
    assert outcome["outcome"] == "inconclusive"
    assert "activation_missing" in outcome["reason"]
    assert "activation" not in outcome
    assert report["negative_outcomes"] == [outcome]


def test_resume_rejects_measured_activation_forged_on_noncomplete_outcome(
    tmp_path: Path,
) -> None:
    class SparseActivation(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            result.pop("mechanism_activated")
            return result

    request = _request(_recipe(max_candidates=1, max_executions=2), output="activation-forged")
    _run_injected(request, base=tmp_path, executor=SparseActivation())
    journal_path = tmp_path / "activation-forged" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["outcomes"][0]["activation"] = {
        "control": True,
        "treatment": True,
        "measured": True,
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "non-complete outcome activation" in resumed.reason
    report = json.loads(
        (tmp_path / "activation-forged" / "experiment-loop-report.json").read_text(encoding="utf-8")
    )
    assert "activation" not in report["outcomes"][0]


def test_resume_does_not_reopen_cancellation_when_ceiling_widens(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=2, max_executions=4)
    first_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "immutable-cancel",
        {
            "recipe": recipe,
            "max_candidates": 1,
            "max_executions": 2,
            "cancel_requested": True,
        },
    )
    first_executor = FakeExecutor()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "cancelled"
    assert first_executor.calls == []

    resumed_request = ComponentRequest(
        "loop-request",
        COMPONENT_ID,
        (SourceRef("loop-source", "source.json", "fixture.v1"),),
        "immutable-cancel",
        {"recipe": recipe, "max_candidates": 2, "max_executions": 4},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "cancelled"
    assert resumed.reason == "cancellation_requested"
    assert resumed_executor.calls == []


def test_resume_answerability_is_authoritative_to_request_config(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2), output="answerability")
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "answerability" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["answerability"] = {
        "schema_version": "research_answerability.v1",
        "state": "answerable",
        "decision_capable": True,
        "reasons": [],
        "warnings": [],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "answerability" in resumed.reason
    report = json.loads(
        (tmp_path / "answerability" / "experiment-loop-report.json").read_text(encoding="utf-8")
    )
    assert report["answerability"] is None


def test_unsupported_recipe_and_failed_control_fidelity_are_truthful(tmp_path: Path) -> None:
    unsupported = _recipe(
        [{"intervention_id": "bad", "factor": "unsupported-factor", "priority": 1}],
        max_candidates=1,
        max_executions=2,
    )
    fake = FakeExecutor()
    result = _run_injected(
        _request(unsupported),
        base=tmp_path,
        executor=fake,
    )
    assert result.status == "unavailable"
    assert fake.calls == []

    class BadControl(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            if kind == "control":
                result["fidelity"] = False
            return result

    fake = BadControl()
    result = _run_injected(
        _request(_recipe(max_candidates=1, max_executions=2), output="fidelity"),
        base=tmp_path,
        executor=fake,
    )
    assert result.status == "failed"
    assert "control_fidelity_failure" in result.reason
    assert [call["kind"] for call in fake.calls] == ["control"]


def test_crash_after_dispatch_recovers_by_operation_id_without_duplicate(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)

    class CrashAfterDispatch(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            self.calls.append({"operation_id": operation_id, "kind": kind})
            result = {
                "status": "ok",
                "metrics": _metrics(kind),
                "mechanism_activated": True,
            }
            self.results[operation_id] = result
            raise KeyboardInterrupt("simulated crash after dispatch")

    crashing = CrashAfterDispatch()
    with pytest.raises(KeyboardInterrupt):
        _run_injected(
            request,
            base=tmp_path,
            executor=crashing,
        )
    recovering = FakeExecutor()
    recovering.results.update(crashing.results)
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=recovering,
    )
    assert resumed.status == "complete"
    assert [call["kind"] for call in recovering.calls] == ["treatment"]
    assert len({call["operation_id"] for call in crashing.calls}) == 1


def test_crash_after_control_settlement_reconciles_pair_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-control crash resumes treatment without rerunning control."""

    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe, output="control-settlement-crash")
    original_control_fidelity = review_experiment_loop._control_fidelity

    def crash_after_control_settlement(
        result: dict[str, Any], *, motion_epsilon: float
    ) -> tuple[bool, str]:
        del result, motion_epsilon
        raise KeyboardInterrupt("simulated crash after control settlement")

    monkeypatch.setattr(
        review_experiment_loop,
        "_control_fidelity",
        crash_after_control_settlement,
    )
    with pytest.raises(KeyboardInterrupt):
        _run_injected(request, base=tmp_path, executor=FakeExecutor())

    journal_path = tmp_path / "control-settlement-crash" / SESSION_JOURNAL_FILENAME
    crashed_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert crashed_journal["executions_consumed"] == 1
    assert crashed_journal["reserved_executions"] == 2
    assert crashed_journal["candidates"]["high"]["reservation"] == 2
    assert [operation["kind"] for operation in crashed_journal["operations"]] == ["control"]
    assert crashed_journal["operations"][0]["state"] == "completed"

    monkeypatch.setattr(review_experiment_loop, "_control_fidelity", original_control_fidelity)
    recovering = FakeExecutor()
    resumed = _run_injected(request, base=tmp_path, executor=recovering, resume=True)

    assert resumed.status == "complete", resumed.reason
    assert [(call["kind"], call["attempt"]) for call in recovering.calls] == [("treatment", 1)]
    resumed_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert resumed_journal["executions_consumed"] == 2
    assert resumed_journal["reserved_executions"] == 0
    assert resumed_journal["accounting"]["controls"] == 1
    assert resumed_journal["accounting"]["treatments"] == 1
    assert resumed_journal["candidates"]["high"]["state"] == "complete"


def test_crash_before_dispatch_fails_closed_without_replaying_operation(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)

    class CrashBeforeDispatch(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            del candidate, kind, spec, attempt
            self.calls.append({"operation_id": operation_id, "crash": "before-dispatch"})
            raise KeyboardInterrupt("simulated crash before dispatch")

    crashing = CrashBeforeDispatch()
    with pytest.raises(KeyboardInterrupt):
        _run_injected(
            request,
            base=tmp_path,
            executor=crashing,
        )
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "crash_recovery_unknown" in resumed.reason or "failed" in resumed.reason


def test_repeated_resume_is_idempotent_and_tampered_source_fails_closed(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    first_executor = FakeExecutor()
    first = _run_injected(
        request,
        base=tmp_path,
        executor=first_executor,
    )
    assert first.status == "complete"
    second_executor = FakeExecutor()
    second = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=second_executor,
    )
    assert second.status == "complete"
    assert second_executor.calls == []

    journal_path = tmp_path / "loop" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    journal["source_admission"]["receipt_id"] = "tampered"
    journal_path.write_text(json.dumps(journal))
    tampered = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert tampered.status == "failed"
    assert "source_admission" in tampered.reason or "journal" in tampered.reason


def test_direct_loop_rejects_existing_journal_without_resume(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    path = tmp_path / "journal.json"
    kwargs = {
        "request": request,
        "recipe": recipe,
        "budget": LoopBudget(max_candidates=1, max_executions=2, wall_timeout_s=600),
        "policy": LoopPolicy(autonomous=True),
        "journal_path": path,
        "executor": FakeExecutor(),
        "source_admission": _source_proof(tmp_path, request),
    }
    ExperimentLoop(**kwargs)
    with pytest.raises(ExperimentLoopError, match="output_collision"):
        ExperimentLoop(**kwargs)
    assert path.exists()
    assert path.with_name(LEGACY_SESSION_JOURNAL_FILENAME).exists()


def test_injected_admission_rejects_status_only_and_host_source(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2))
    fake = FakeExecutor()
    status_only = run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission={"status": "admitted"},
    )
    assert status_only.status == "unavailable"
    assert fake.calls == []

    proof = _source_proof(tmp_path, request)
    proof["source_root"] = "/etc"
    proof["source"] = {
        **proof["source"],
        "uri": "/etc/passwd",
    }
    host_source = run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=proof,
    )
    assert host_source.status == "unavailable"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda journal: journal.update({"status": "complete", "outcomes": []}),
        lambda journal: journal.update({"status": "partial"}),
        lambda journal: journal.update({"executions_consumed": 0}),
        lambda journal: journal["candidates"]["high"].update({"state": "pending"}),
        lambda journal: journal["operations"][0].update(
            {"state": "completed", "result": {"status": "failed"}}
        ),
    ],
    ids=["terminal-outcomes", "partial-reason", "consumed", "candidate-state", "operation-state"],
)
def test_resume_rejects_forged_terminal_journal(tmp_path: Path, mutation: Any) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "loop" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    mutation(journal)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "cannot resume" in resumed.reason


def _omit_treatment_from_complete_journal(journal: dict[str, Any]) -> None:
    treatment = next(
        operation for operation in journal["operations"] if operation["kind"] == "treatment"
    )
    journal["operations"].remove(treatment)
    candidate = journal["candidates"]["high"]
    candidate["operation_ids"].remove(treatment["operation_id"])
    candidate["attempts"] = 1
    journal["accounting"]["treatments"] = 0
    journal["executions_consumed"] = 1
    journal["outcomes"][0]["operation_ids"] = list(candidate["operation_ids"])


def _forge_status_only_treatment_result(journal: dict[str, Any]) -> None:
    treatment = next(
        operation for operation in journal["operations"] if operation["kind"] == "treatment"
    )
    treatment["result"] = {"status": "ok"}


def _forge_pair_verdict_from_retained_telemetry(journal: dict[str, Any]) -> None:
    outcome = journal["outcomes"][0]
    assert outcome["outcome"] == "survived"
    outcome["outcome"] = "falsified"
    outcome["verdict"] = "falsified"
    outcome["negative"] = True


@pytest.mark.parametrize(
    "mutation",
    [
        _omit_treatment_from_complete_journal,
        _forge_status_only_treatment_result,
        _forge_pair_verdict_from_retained_telemetry,
    ],
    ids=["missing-treatment-pair", "status-only-treatment", "telemetry-verdict-mismatch"],
)
def test_resume_rejects_forged_complete_pair_journal(tmp_path: Path, mutation: Any) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "loop" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    mutation(journal)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "cannot resume" in resumed.reason


def test_native_adapter_dispatches_one_pair_per_candidate_and_honors_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(max_candidates=2, max_executions=4)
    request = _request(recipe)
    child_calls: list[tuple[int, bool]] = []

    def fake_child_run(
        child_request: ComponentRequest,
        *,
        base: Path,
        resume: bool,
        admission_config: Any,
    ) -> ComponentResult:
        del admission_config
        selected = sorted(
            child_request.config["recipe"]["interventions"],
            key=lambda item: (item["priority"], item["intervention_id"]),
        )[: child_request.config["max_candidates"]]
        child_calls.append((len(selected), resume))
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        reports = [
            {
                "intervention_id": item["intervention_id"],
                "factor": item["factor"],
                "status": "complete",
                "control_metrics": _metrics("control"),
                "treatment_metrics": _metrics("treatment", value=1.2),
                "control_activated": True,
                "treatment_activated": True,
            }
            for item in selected
        ]
        (child_dir / "execute-report.json").write_text(
            json.dumps({"candidates": reports}), encoding="utf-8"
        )
        return ComponentResult(
            request_id=child_request.request_id,
            component_id="srev22-review-execute",
            status="complete",
        )

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        fake_child_run,
    )
    cancelled = False
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 2, "max_executions": 4},
        recipe=recipe,
        admission_config={},
        resume=False,
        cancel=lambda: cancelled,
    )
    first = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[0]
    second = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[1]
    assert adapter.execute(operation_id="one", candidate=first, kind="control")["status"] == "ok"
    assert adapter.execute(operation_id="two", candidate=first, kind="treatment")["status"] == "ok"
    cancelled = True
    assert (
        adapter.execute(operation_id="three", candidate=second, kind="control")["status"]
        == "cancelled"
    )
    assert child_calls == [(1, False)]


def test_native_child_failure_precedes_retained_candidate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)

    def fake_child_run(*args: Any, **kwargs: Any) -> ComponentResult:
        base = kwargs["base"]
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "candidates": [
                {
                    "intervention_id": "high",
                    "factor": "single_pedestrian_speed_offset",
                    "status": "complete",
                    "control_metrics": _metrics("control"),
                    "treatment_metrics": _metrics("treatment"),
                }
            ]
        }
        (child_dir / "execute-report.json").write_text(json.dumps(report), encoding="utf-8")
        return ComponentResult(
            request_id=args[0].request_id,
            component_id="srev22-review-execute",
            status="failed",
            reason="source admission failed",
        )

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        fake_child_run,
    )
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 1, "max_executions": 2},
        recipe=recipe,
        admission_config={},
        resume=False,
    )
    candidate = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[0]
    assert adapter.execute(operation_id="one", candidate=candidate, kind="control") == {
        "status": "failed",
        "reason": "source admission failed",
    }


def test_native_stale_execute_report_cannot_mask_failed_attempt_ledger(
    tmp_path: Path,
) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    child_dir = tmp_path / "executor"
    child_dir.mkdir(parents=True, exist_ok=True)
    stale_report = {
        "candidates": [
            {
                "intervention_id": "high",
                "factor": "single_pedestrian_speed_offset",
                "status": "complete",
                "control_metrics": _metrics("control"),
                "treatment_metrics": _metrics("treatment"),
            }
        ]
    }
    failed_ledger = {
        "candidate_reports": [
            {
                "intervention_id": "high",
                "factor": "single_pedestrian_speed_offset",
                "status": "failed",
                "reason": "control execution failed: child attempt failed",
            }
        ],
        "attempts": [
            {
                "candidate_id": "high",
                "kind": "control",
                "status": "failed",
                "reason": "child attempt failed",
            }
        ],
    }
    (child_dir / "execute-report.json").write_text(json.dumps(stale_report), encoding="utf-8")
    (child_dir / "attempt-ledger.json").write_text(json.dumps(failed_ledger), encoding="utf-8")
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 1, "max_executions": 2},
        recipe=recipe,
        admission_config={},
        resume=True,
    )
    candidate = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[0]
    result = adapter.execute(operation_id="one", candidate=candidate, kind="control")
    assert result["status"] == "failed"
    assert "disagrees" in result["reason"]


def test_native_control_fidelity_failure_counts_child_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, admission = _native_fixture_request("native-fidelity")

    def fake_child_run(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        del kwargs
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "candidates": [
                {
                    "intervention_id": "ped-speed-up",
                    "factor": "single_pedestrian_speed_offset",
                    "status": "failed",
                    "reason": "control_fidelity_failure: synthetic child check",
                    "control_metrics": _metrics("control"),
                }
            ]
        }
        (child_dir / "execute-report.json").write_text(json.dumps(report), encoding="utf-8")
        return ComponentResult(
            request_id=child_request.request_id,
            component_id="srev22-review-execute",
            status="failed",
            reason="candidate_execution_failed: control_fidelity_failure: synthetic child check",
        )

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        fake_child_run,
    )
    result = run(request, base=tmp_path, autonomous=True, admission_config=admission)
    assert result.status == "failed"
    journal = json.loads((tmp_path / "native-fidelity" / SESSION_JOURNAL_FILENAME).read_text())
    assert journal["accounting"] == {
        "controls": 1,
        "treatments": 0,
        "failures": 0,
        "retries": 0,
        "fidelity_attempts": 2,
    }, result.reason


def test_native_treatment_failure_preserves_treatment_side_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, admission = _native_fixture_request("native-treatment-failure")

    def fake_child_run(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        del kwargs
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "candidates": [
                {
                    "intervention_id": "ped-speed-up",
                    "factor": "single_pedestrian_speed_offset",
                    "status": "failed",
                    "reason": "treatment execution failed: synthetic treatment error",
                    "control_metrics": _metrics("control"),
                }
            ]
        }
        (child_dir / "execute-report.json").write_text(json.dumps(report), encoding="utf-8")
        return ComponentResult(
            request_id=child_request.request_id,
            component_id="srev22-review-execute",
            status="failed",
            reason="candidate_execution_failed: treatment execution failed: synthetic treatment error",
        )

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        fake_child_run,
    )
    result = run(request, base=tmp_path, autonomous=True, admission_config=admission)
    assert result.status == "failed"
    journal = json.loads(
        (tmp_path / "native-treatment-failure" / SESSION_JOURNAL_FILENAME).read_text()
    )
    assert journal["accounting"] == {
        "controls": 1,
        "treatments": 1,
        "failures": 1,
        "retries": 0,
        "fidelity_attempts": 2,
    }
    assert [item["kind"] for item in journal["operations"]] == ["control", "treatment"]


def test_native_cancellation_during_pair_settles_both_outer_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    cancellation = {"requested": False}

    def fake_child_run(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        del kwargs
        cancellation["requested"] = True
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "candidates": [
                {
                    "intervention_id": "high",
                    "factor": "single_pedestrian_speed_offset",
                    "status": "complete",
                    "control_metrics": _metrics("control"),
                    "treatment_metrics": _metrics("treatment"),
                    "control_activated": True,
                    "treatment_activated": True,
                }
            ]
        }
        (child_dir / "execute-report.json").write_text(json.dumps(report), encoding="utf-8")
        return ComponentResult(
            request_id=child_request.request_id,
            component_id="srev22-review-execute",
            status="complete",
        )

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        fake_child_run,
    )
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 1, "max_executions": 2},
        recipe=recipe,
        admission_config={},
        resume=False,
        cancel=lambda: cancellation["requested"],
    )
    candidate = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[0]
    control = adapter.execute(operation_id="control", candidate=candidate, kind="control")
    treatment = adapter.execute(operation_id="treatment", candidate=candidate, kind="treatment")
    assert control["status"] == treatment["status"] == "ok"
    assert control["native_pair_complete"] is True
    assert treatment["native_pair_complete"] is True


def test_native_crash_resume_recovers_nested_pair_before_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, admission = _native_fixture_request("native-crash-resume")
    child_calls: list[str] = []

    def crash_after_nested_pair(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        admission_config = kwargs["admission_config"]
        child_calls.append("nested")
        _write_valid_native_pair(child_request, base, admission_config)
        if len(child_calls) > 1:
            return ComponentResult(
                request_id=child_request.request_id,
                component_id="srev22-review-execute",
                status="complete",
            )
        raise KeyboardInterrupt("crash after nested pair")

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        crash_after_nested_pair,
    )
    with pytest.raises(KeyboardInterrupt):
        run(request, base=tmp_path, autonomous=True, admission_config=admission)

    resumed = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        admission_config=admission,
        cancel=lambda: True,
    )
    assert resumed.status == "cancelled"
    assert child_calls == ["nested", "nested"]
    journal = json.loads((tmp_path / "native-crash-resume" / SESSION_JOURNAL_FILENAME).read_text())
    assert {operation["state"] for operation in journal["operations"]} == {"completed"}
    assert [operation["kind"] for operation in journal["operations"]] == [
        "control",
        "treatment",
    ]
    assert journal["accounting"]["treatments"] == 1
    repeated = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        admission_config=admission,
    )
    assert repeated.status == "cancelled"
    assert child_calls == ["nested", "nested"]


def test_native_crash_before_report_reconciles_both_child_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, admission = _native_fixture_request("native-ledger-pair")
    child_calls: list[str] = []

    def crash_before_report(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        child_calls.append("nested")
        _write_valid_native_pair(child_request, base, kwargs["admission_config"])
        ledger_path = base / "executor" / "attempt-ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["candidate_reports"] = []
        ledger["traces"] = []
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        (base / "executor" / "execute-report.json").unlink()
        raise KeyboardInterrupt("crash before candidate report")

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        crash_before_report,
    )
    with pytest.raises(KeyboardInterrupt):
        run(request, base=tmp_path, autonomous=True, admission_config=admission)

    resumed = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        admission_config=admission,
        cancel=lambda: True,
    )
    assert resumed.status == "cancelled"
    assert child_calls == ["nested"]
    journal = json.loads(
        (tmp_path / "native-ledger-pair" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert [operation["kind"] for operation in journal["operations"]] == [
        "control",
        "treatment",
    ]
    assert {operation["state"] for operation in journal["operations"]} == {"completed"}
    assert journal["executions_consumed"] == 2
    assert journal["accounting"] == {
        "controls": 1,
        "treatments": 1,
        "failures": 0,
        "retries": 0,
        "fidelity_attempts": 2,
    }


def test_native_recovery_rejects_unbound_nested_artifacts_before_child_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, admission = _native_fixture_request("native-unbound-recovery")
    child_calls: list[str] = []

    def crash_after_nested_pair(
        child_request: ComponentRequest, *, base: Path, **kwargs: Any
    ) -> ComponentResult:
        child_calls.append("nested")
        _write_valid_native_pair(child_request, base, kwargs["admission_config"])
        raise KeyboardInterrupt("crash after nested pair")

    monkeypatch.setattr(
        "robot_sf.analysis_workbench.review_experiment_loop.review_execute.run",
        crash_after_nested_pair,
    )
    with pytest.raises(KeyboardInterrupt):
        run(request, base=tmp_path, autonomous=True, admission_config=admission)
    ledger_path = tmp_path / "native-unbound-recovery" / "executor" / "attempt-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["request_digest"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    resumed = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        admission_config=admission,
    )
    assert resumed.status == "failed"
    assert child_calls == ["nested"]
    journal = json.loads(
        (tmp_path / "native-unbound-recovery" / SESSION_JOURNAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert (
        "nested child recovery identity binding mismatch"
        in journal["operations"][0]["result"]["reason"]
    )


def test_native_operation_recovery_uses_exact_map_for_retry_in_candidate_id(
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        [
            {
                "intervention_id": "candidate:retry:literal",
                "factor": "single_pedestrian_speed_offset",
                "priority": 1,
            }
        ],
        max_candidates=1,
        max_executions=2,
    )
    request = _request(recipe)
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 1, "max_executions": 2},
        recipe=recipe,
        admission_config={},
        resume=True,
    )
    candidate_id = "candidate:retry:literal"
    operation_id = _operation_id("session", candidate_id, "control")
    adapter.bind_operation_map(
        [{"operation_id": operation_id, "candidate_id": candidate_id, "kind": "control"}]
    )
    adapter.child_result = ComponentResult(
        request_id=request.request_id,
        component_id="srev22-review-execute",
        status="complete",
    )
    adapter.reports[candidate_id] = {
        "intervention_id": candidate_id,
        "status": "complete",
        "control_metrics": _metrics("control"),
        "treatment_metrics": _metrics("treatment"),
        "control_activated": True,
        "treatment_activated": True,
    }
    assert adapter.result_for(operation_id)["status"] == "ok"
    assert adapter.result_for(operation_id.replace(candidate_id, "literal")) is None


def test_native_report_without_activation_does_not_infer_from_metrics(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    adapter = _NativeExecutorAdapter(
        request,
        base=tmp_path,
        executor_config={"max_candidates": 1, "max_executions": 2},
        recipe=recipe,
        admission_config={},
        resume=False,
    )
    candidate = sorted(
        recipe["interventions"], key=lambda item: (item["priority"], item["intervention_id"])
    )[0]
    adapter.child_result = ComponentResult(
        request_id=request.request_id,
        component_id="srev22-review-execute",
        status="complete",
    )
    adapter.reports[candidate["intervention_id"]] = {
        "intervention_id": candidate["intervention_id"],
        "status": "complete",
        "control_metrics": _metrics("control"),
        "treatment_metrics": _metrics("treatment"),
    }
    result = adapter.execute(
        operation_id="control",
        candidate=candidate,
        kind="control",
    )
    assert result["status"] == "unavailable"
    assert "activation telemetry" in result["reason"]


def test_resume_rejects_reordered_self_consistent_operation_list(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2), output="operation-order")
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "operation-order" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["operations"].reverse()
    for sequence, operation in enumerate(journal["operations"]):
        operation["sequence"] = sequence
    candidate = journal["candidates"]["high"]
    candidate["operation_ids"] = [operation["operation_id"] for operation in journal["operations"]]
    journal["outcomes"][0]["operation_ids"] = list(candidate["operation_ids"])
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "treatment operation lacks prior control sequence" in resumed.reason


def test_wall_deadline_between_control_and_treatment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeMonotonicClock()
    monkeypatch.setattr(time, "monotonic", clock)

    class SlowControl(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ):
            if kind == "control":
                clock.advance(1.0)
            return super().execute(operation_id, candidate, kind, spec, attempt)

    request = _request(_recipe(max_candidates=1, max_executions=2))
    request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "wall_timeout_s": 0.5},
    )
    executor = SlowControl()
    result = _run_injected(request, base=tmp_path, executor=executor)
    assert result.status == "partial"
    assert result.reason == "wall_timeout"
    assert [call["kind"] for call in executor.calls] == ["control"]


def test_wall_deadline_resume_dispatches_remaining_treatment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeMonotonicClock()
    monkeypatch.setattr(time, "monotonic", clock)

    class SlowControl(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ):
            if kind == "control":
                clock.advance(1.0)
            return super().execute(operation_id, candidate, kind, spec, attempt)

    request = _request(_recipe(max_candidates=1, max_executions=2), output="wall-resume")
    first_request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "wall_timeout_s": 0.5},
    )
    first_executor = SlowControl()
    first = _run_injected(first_request, base=tmp_path, executor=first_executor)
    assert first.status == "partial"
    assert first.reason == "wall_timeout"
    assert [call["kind"] for call in first_executor.calls] == ["control"]

    resumed_request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "wall_timeout_s": 2.0},
    )
    resumed_executor = FakeExecutor()
    resumed = _run_injected(
        resumed_request,
        base=tmp_path,
        resume=True,
        executor=resumed_executor,
    )
    assert resumed.status == "complete", resumed.reason
    assert [call["kind"] for call in resumed_executor.calls] == ["treatment"]
    journal = json.loads(
        (tmp_path / "wall-resume" / SESSION_JOURNAL_FILENAME).read_text(encoding="utf-8")
    )
    assert journal["executions_consumed"] == 2
    assert journal["accounting"]["controls"] == 1
    assert journal["accounting"]["treatments"] == 1


def test_post_result_cancellation_precedes_recipe_terminal_marker(tmp_path: Path) -> None:
    cancellation = {"requested": False}

    class TerminalTreatment(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            result["terminal"] = True
            if kind == "treatment":
                cancellation["requested"] = True
            return result

    request = _request(_recipe(max_candidates=2, max_executions=4), output="post-cancel")
    executor = TerminalTreatment()
    result = _run_injected(
        request,
        base=tmp_path,
        executor=executor,
        cancel=lambda: cancellation["requested"],
    )
    assert result.status == "cancelled"
    assert result.reason == "cancellation_requested"
    assert [call["candidate"] for call in executor.calls] == ["high", "high"]


def test_post_result_wall_deadline_precedes_recipe_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeMonotonicClock()
    monkeypatch.setattr(time, "monotonic", clock)

    class TerminalTreatment(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            result["terminal"] = True
            if kind == "treatment":
                clock.advance(1.0)
            return result

    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe, output="post-deadline")
    request = ComponentRequest(
        request.request_id,
        request.component_id,
        request.sources,
        request.output_directory,
        {**request.config, "wall_timeout_s": 0.5},
    )
    executor = TerminalTreatment()
    result = _run_injected(request, base=tmp_path, executor=executor)
    assert result.status == "partial"
    assert result.reason == "wall_timeout"
    assert [call["kind"] for call in executor.calls] == ["control", "treatment"]


def test_resume_rejects_lowered_persisted_elapsed_deadline(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2), output="elapsed-floor")
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "elapsed-floor" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    original_elapsed = float(journal["elapsed_s"])
    # Keep the mutation below the persisted floor even when the fast test run
    # rounds the first journal write to zero seconds.
    journal["elapsed_s"] = max(0.0, original_elapsed - 0.001)
    if journal["elapsed_s"] >= float(journal["elapsed_floor_s"]):
        journal["elapsed_floor_s"] = max(float(journal["elapsed_floor_s"]), 0.001)
        journal["elapsed_s"] = 0.0
    assert journal["elapsed_s"] < float(journal["elapsed_floor_s"])
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "elapsed accounting" in resumed.reason


def test_resume_rejects_missing_persisted_elapsed_floor(tmp_path: Path) -> None:
    request = _request(_recipe(max_candidates=1, max_executions=2), output="missing-floor")
    first = _run_injected(request, base=tmp_path, executor=FakeExecutor())
    assert first.status == "complete"
    journal_path = tmp_path / "missing-floor" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    del journal["elapsed_floor_s"]
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed = _run_injected(
        request,
        base=tmp_path,
        resume=True,
        executor=FakeExecutor(),
    )
    assert resumed.status == "failed"
    assert "elapsed accounting floor is missing" in resumed.reason


def test_session_lock_rejects_concurrent_resume(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class Blocking(FakeExecutor):
        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ):
            started.set()
            release.wait(timeout=5)
            return super().execute(operation_id, candidate, kind, spec, attempt)

    request = _request(_recipe(max_candidates=1, max_executions=2))
    first_result: list[ComponentResult] = []

    def first_controller() -> None:
        try:
            first_result.append(_run_injected(request, base=tmp_path, executor=Blocking()))
        finally:
            completed.set()

    thread = threading.Thread(target=first_controller)
    thread.start()
    try:
        assert started.wait(timeout=5)
        second = _run_injected(request, base=tmp_path, resume=True, executor=FakeExecutor())
        assert second.status == "failed"
        assert "session_lock_owned" in second.reason
    finally:
        release.set()
        finished = completed.wait(timeout=30)
        thread.join(timeout=1)

    assert finished, "original controller did not complete after releasing the session lock"
    assert not thread.is_alive()
    assert first_result and first_result[0].status == "complete"


def test_real_admitted_fixture_smoke_uses_native_control_and_treatment(tmp_path: Path) -> None:
    """Exercise one supported source-bound pair through the real SREV-22 path."""

    fixture_root = Path("tests/fixtures/scenario_review/review_experiment_loop")
    request_payload = json.loads((fixture_root / "request.json").read_text())
    config = json.loads((fixture_root / "config.json").read_text())
    admission = json.loads((fixture_root / "admission.json").read_text())
    request_payload["config"] = {
        **config,
        "max_candidates": 1,
        "max_executions": 2,
        "executor_config": {**config["executor_config"], "horizon_steps": 1},
    }
    request_payload["output_directory"] = "native-smoke"
    request = ComponentRequest(
        request_payload["request_id"],
        request_payload["component_id"],
        (SourceRef(**request_payload["sources"][0]),),
        request_payload["output_directory"],
        request_payload["config"],
    )
    result = run(
        request,
        base=tmp_path,
        autonomous=True,
        admission_config=admission,
    )
    assert result.status == "complete"
    report = json.loads((tmp_path / "native-smoke" / "experiment-loop-report.json").read_text())
    ledger = json.loads(
        (tmp_path / "native-smoke" / "executor" / "attempt-ledger.json").read_text()
    )
    assert report["source_admission"]["receipt_id"] == "srev22-review-execute-receipt"
    assert report["provenance"]["scientific_claim_allowed"] is False
    assert report["budget"]["executions_consumed"] == 2
    assert ledger["executions_consumed"] == report["budget"]["executions_consumed"]
