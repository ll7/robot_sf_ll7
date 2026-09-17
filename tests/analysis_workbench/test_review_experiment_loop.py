"""Focused SREV-24 loop contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import ComponentRequest, SourceRef
from robot_sf.analysis_workbench.review_experiment_loop import (
    COMPONENT_ID,
    LEGACY_SESSION_JOURNAL_FILENAME,
    SESSION_JOURNAL_FILENAME,
    ExperimentLoop,
    ExperimentLoopError,
    LoopBudget,
    LoopPolicy,
    descriptor,
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
        }
        self.results[operation_id] = result
        return result

    def result_for(self, operation_id: str) -> dict[str, Any] | None:
        return self.results.get(operation_id)


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
    result = run(
        _request(recipe),
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission={"status": "admitted", "receipt_id": "fixture"},
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


def test_pair_reservation_prevents_partial_new_pair(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=2, max_executions=3)
    fake = FakeExecutor()
    result = run(
        _request(recipe),
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission={"status": "admitted"},
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
    first = run(
        first_request,
        base=tmp_path,
        autonomous=True,
        executor=first_executor,
        source_admission={"status": "admitted"},
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
    resumed = run(
        resumed_request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=resumed_executor,
        source_admission={"status": "admitted"},
    )
    assert resumed.status == "complete"
    assert [call["candidate"] for call in resumed_executor.calls] == ["tie-id", "tie-id"]
    journal = json.loads((tmp_path / "loop" / SESSION_JOURNAL_FILENAME).read_text())
    assert journal["executions_consumed"] == 4


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
    result = run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=executor,
        source_admission={"status": "admitted"},
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


def test_cancellation_after_control_does_not_dispatch_treatment(tmp_path: Path) -> None:
    executor = FakeExecutor()
    result = run(
        _request(_recipe(max_candidates=1, max_executions=2)),
        base=tmp_path,
        autonomous=True,
        executor=executor,
        source_admission={"status": "admitted"},
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

    result = run(
        _request(recipe),
        base=tmp_path,
        autonomous=True,
        executor=Outcomes(),
        source_admission={"status": "admitted"},
    )
    assert result.status == "complete"
    report = json.loads((tmp_path / "loop" / "experiment-loop-report.json").read_text())
    assert [item["outcome"] for item in report["outcomes"]] == ["falsified", "inconclusive"]
    assert len(report["negative_outcomes"]) == 2


def test_unsupported_recipe_and_failed_control_fidelity_are_truthful(tmp_path: Path) -> None:
    unsupported = _recipe(
        [{"intervention_id": "bad", "factor": "unsupported-factor", "priority": 1}],
        max_candidates=1,
        max_executions=2,
    )
    fake = FakeExecutor()
    result = run(
        _request(unsupported),
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission={"status": "admitted"},
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
    result = run(
        _request(_recipe(max_candidates=1, max_executions=2), output="fidelity"),
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission={"status": "admitted"},
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
            result = {"status": "ok", "metrics": _metrics(kind)}
            self.results[operation_id] = result
            raise KeyboardInterrupt("simulated crash after dispatch")

    crashing = CrashAfterDispatch()
    with pytest.raises(KeyboardInterrupt):
        run(
            request,
            base=tmp_path,
            autonomous=True,
            executor=crashing,
            source_admission={"status": "admitted"},
        )
    recovering = FakeExecutor()
    recovering.results.update(crashing.results)
    resumed = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=recovering,
        source_admission={"status": "admitted"},
    )
    assert resumed.status == "complete"
    assert [call["kind"] for call in recovering.calls] == ["treatment"]
    assert len({call["operation_id"] for call in crashing.calls}) == 1


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
        run(
            request,
            base=tmp_path,
            autonomous=True,
            executor=crashing,
            source_admission={"status": "admitted"},
        )
    resumed = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=FakeExecutor(),
        source_admission={"status": "admitted"},
    )
    assert resumed.status == "failed"
    assert "crash_recovery_unknown" in resumed.reason or "failed" in resumed.reason


def test_repeated_resume_is_idempotent_and_tampered_source_fails_closed(tmp_path: Path) -> None:
    recipe = _recipe(max_candidates=1, max_executions=2)
    request = _request(recipe)
    first_executor = FakeExecutor()
    first = run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=first_executor,
        source_admission={"status": "admitted", "receipt_id": "one"},
    )
    assert first.status == "complete"
    second_executor = FakeExecutor()
    second = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=second_executor,
        source_admission={"status": "admitted", "receipt_id": "one"},
    )
    assert second.status == "complete"
    assert second_executor.calls == []

    journal_path = tmp_path / "loop" / SESSION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    journal["source_admission"]["receipt_id"] = "tampered"
    journal_path.write_text(json.dumps(journal))
    tampered = run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=FakeExecutor(),
        source_admission={"status": "admitted", "receipt_id": "one"},
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
        "source_admission": {"status": "admitted"},
    }
    ExperimentLoop(**kwargs)
    with pytest.raises(ExperimentLoopError, match="output_collision"):
        ExperimentLoop(**kwargs)
    assert path.exists()
    assert path.with_name(LEGACY_SESSION_JOURNAL_FILENAME).exists()


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
    assert report["source_admission"]["receipt_id"] == "srev22-review-execute-receipt"
    assert report["provenance"]["scientific_claim_allowed"] is False
    assert report["budget"]["executions_consumed"] == 2
