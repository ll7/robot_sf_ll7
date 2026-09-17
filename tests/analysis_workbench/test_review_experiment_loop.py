"""Focused SREV-24 loop contract tests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

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
    _canonical_digest,
    _NativeExecutorAdapter,
    _operation_id,
    _request_identity,
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
        }
        self.results[operation_id] = result
        return result

    def result_for(self, operation_id: str) -> dict[str, Any] | None:
        return self.results.get(operation_id)


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
            result = {"status": "ok", "metrics": _metrics(kind)}
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


@pytest.mark.parametrize(
    "mutation",
    [_omit_treatment_from_complete_journal, _forge_status_only_treatment_result],
    ids=["missing-treatment-pair", "status-only-treatment"],
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
        del kwargs
        child_calls.append("nested")
        child_dir = base / "executor"
        child_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "candidates": [
                {
                    "intervention_id": "ped-speed-up",
                    "factor": "single_pedestrian_speed_offset",
                    "status": "complete",
                    "control_metrics": _metrics("control"),
                    "treatment_metrics": _metrics("treatment", value=1.2),
                    "control_activated": True,
                    "treatment_activated": True,
                }
            ]
        }
        (child_dir / "execute-report.json").write_text(json.dumps(report), encoding="utf-8")
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
    assert child_calls == ["nested"]
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
    assert child_calls == ["nested"]


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
    }
    assert adapter.result_for(operation_id)["status"] == "ok"
    assert adapter.result_for(operation_id.replace(candidate_id, "literal")) is None


def test_wall_deadline_between_control_and_treatment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_monotonic = time.monotonic
    clock = {"jump": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + clock["jump"])

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
                clock["jump"] = 1.0
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


def test_session_lock_rejects_concurrent_resume(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

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
        first_result.append(_run_injected(request, base=tmp_path, executor=Blocking()))

    thread = threading.Thread(target=first_controller)
    thread.start()
    assert started.wait(timeout=5)
    second = _run_injected(request, base=tmp_path, resume=True, executor=FakeExecutor())
    assert second.status == "failed"
    assert "session_lock_owned" in second.reason
    release.set()
    thread.join(timeout=5)
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
